# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Modifications Copyright 2025 CRRL authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CRRL Trainer extending PPO Trainer with Self-Play Optimization.
This trainer inherits from the base PPO trainer and adds CRRL-specific logic.
"""

import json
import os
import re
import shutil
import hashlib
import math
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as BaseRayPPOTrainer
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

# Re-export for backward compatibility
__all__ = [
    "RayPPOTrainer",
    "ResourcePoolManager",
    "Role",
    "apply_kl_penalty",
    "compute_advantage",
    "compute_response_mask",
]


class RayPPOTrainer(BaseRayPPOTrainer):
    """CRRL-specific PPO trainer that extends the base trainer with Self-Play Optimization logic.

    This trainer inherits most functionality from the base RayPPOTrainer and adds:
    - CRRL-specific weighted sampling based on Thompson sampling
    - CRRL advantage calculation with external baseline or value estimation
    """

    _THINK_CONTENT_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
    _ANSWER_CONTENT_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
    _ESTIMATOR_RESUME_STATE_FILENAME = "crrl_adaptive_estimator_resume_state.pt"
    _ESTIMATOR_RESUME_META_FILENAME = "crrl_adaptive_estimator_resume_state.meta.json"
    _ESTIMATOR_SNAPSHOT_FILENAME = "crrl_adaptive_estimator_current.joblib"
    _ESTIMATOR_UPDATE_ROOT_DIRNAME = "adaptive_estimator_updates"
    _ESTIMATOR_UPDATE_META_FILENAME = "crrl_adaptive_estimator_update.meta.json"
    _ESTIMATOR_LATEST_UPDATE_FILENAME = "latest_adaptive_estimator_update.txt"
    _PROMPT_REWARD_LOG_ROOT_DIRNAME = "prompt_reward_logs"
    _PROMPT_REWARD_LATEST_FILENAME = "latest_prompt_reward_log.txt"

    def _default_local_ckpt_root(self) -> str:
        checkpoint_folder = str(self.config.trainer.default_local_dir)
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        return checkpoint_folder

    def _current_global_step_ckpt_dir(self) -> str:
        return os.path.join(self._default_local_ckpt_root(), f"global_step_{self.global_steps}")

    def _default_estimator_update_root(self) -> str:
        return os.path.join(self._default_local_ckpt_root(), self._ESTIMATOR_UPDATE_ROOT_DIRNAME)

    def _default_prompt_reward_log_root(self) -> str:
        return os.path.join(self._default_local_ckpt_root(), self._PROMPT_REWARD_LOG_ROOT_DIRNAME)

    def _resolve_estimator_update_output_dir(self, output_dir: str | None) -> str:
        if output_dir is not None:
            path = str(output_dir).strip()
            if path and path.lower() not in {"none", "null", "auto"}:
                path = os.path.expanduser(path)
                if not os.path.isabs(path):
                    path = os.path.join(os.getcwd(), path)
                return path
        return self._default_estimator_update_root()

    def _resolve_prompt_reward_log_dir(self, output_dir: str | None) -> str:
        if output_dir is not None:
            path = str(output_dir).strip()
            if path and path.lower() not in {"none", "null", "auto"}:
                path = os.path.expanduser(path)
                if not os.path.isabs(path):
                    path = os.path.join(os.getcwd(), path)
                return path
        return self._default_prompt_reward_log_root()

    @staticmethod
    def _config_bool(value, *, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
            raise ValueError(f"{name} must be a boolean value, got: {value!r}")
        return bool(value)

    @staticmethod
    def _new_estimator_row_buffer() -> dict[str, list]:
        return {
            "prompt_hidden_rows": [],
            "response_hidden_rows": [],
            "response_feature_rows": [],
            "targets": [],
        }

    @staticmethod
    def _extend_estimator_row_buffer(dst: dict[str, list], src: dict[str, list]) -> None:
        for key in ("prompt_hidden_rows", "response_hidden_rows", "response_feature_rows", "targets"):
            dst[key].extend(src[key])

    @staticmethod
    def _normalize_step_row_counts(values: list | np.ndarray | None) -> list[int]:
        if values is None:
            return []
        return [int(v) for v in values if int(v) > 0]

    @staticmethod
    def _normalize_hidden_rows(rows: list | None) -> list[np.ndarray]:
        if rows is None:
            return []
        normalized_rows: list[np.ndarray] = []
        for row in rows:
            normalized_rows.append(np.asarray(row, dtype=np.float32).reshape(-1))
        return normalized_rows

    @staticmethod
    def _normalize_feature_rows(rows: list | None) -> list[dict[str, float]]:
        if rows is None:
            return []
        normalized_rows: list[dict[str, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError(f"Invalid estimator response feature row type: {type(row)}")
            normalized_rows.append({str(k): float(v) for k, v in row.items()})
        return normalized_rows

    @staticmethod
    def _normalize_targets(values: list | np.ndarray | None) -> list[float]:
        if values is None:
            return []
        return [float(v) for v in values]

    @staticmethod
    def _build_stable_uids_from_input_ids(input_ids: torch.Tensor) -> np.ndarray:
        uids = []
        for row in input_ids:
            uid_str = hashlib.md5(str(row.tolist()).encode("utf-8")).hexdigest()
            uids.append(uid_str)
        return np.array(uids, dtype=object)

    @staticmethod
    def _get_nonzero_advantage_indices(
        batch: DataProto, *, zero_eps: float, raw_advantages: torch.Tensor | None = None
    ) -> tuple[list[int], int, int]:
        if raw_advantages is None and "advantages" not in batch.batch:
            raise KeyError(
                "Group filtering requires either `raw_advantages` input or 'advantages' in batch.batch."
            )
        if "uid" not in batch.non_tensor_batch:
            raise KeyError("Group filtering requires non_tensor field 'uid'.")

        if raw_advantages is not None:
            if raw_advantages.ndim == 1:
                seq_adv = raw_advantages.to(torch.float32)
            elif raw_advantages.ndim == 2:
                response_mask = batch.batch.get("response_mask", None)
                if response_mask is not None and response_mask.shape == raw_advantages.shape:
                    mask = response_mask.to(torch.float32)
                    denom = mask.sum(dim=-1).clamp_min(1.0)
                    seq_adv = (raw_advantages.to(torch.float32) * mask).sum(dim=-1) / denom
                else:
                    seq_adv = raw_advantages.to(torch.float32).mean(dim=-1)
            else:
                raise ValueError(
                    f"Unsupported raw_advantages rank={raw_advantages.ndim}. Expected rank 1 or 2."
                )
        else:
            advantages = batch.batch["advantages"]
            response_mask = batch.batch.get("response_mask", None)
            if advantages.ndim == 1:
                seq_adv = advantages.to(torch.float32)
            elif response_mask is not None and response_mask.shape == advantages.shape:
                mask = response_mask.to(torch.float32)
                denom = mask.sum(dim=-1).clamp_min(1.0)
                seq_adv = (advantages.to(torch.float32) * mask).sum(dim=-1) / denom
            else:
                seq_adv = advantages.to(torch.float32).mean(dim=-1)

        seq_adv_np = seq_adv.detach().cpu().numpy().astype(np.float32, copy=False)
        uid_abs_max_adv: dict[str, float] = {}
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            uid_str = str(uid)
            abs_adv = float(abs(seq_adv_np[idx]))
            if not np.isfinite(abs_adv):
                abs_adv = 0.0
            prev = uid_abs_max_adv.get(uid_str)
            if prev is None or abs_adv > prev:
                uid_abs_max_adv[uid_str] = abs_adv

        kept_prompt_uids = {
            uid for uid, max_abs_adv in uid_abs_max_adv.items() if max_abs_adv > zero_eps
        }
        kept_indices = [
            idx for idx, uid in enumerate(batch.non_tensor_batch["uid"]) if str(uid) in kept_prompt_uids
        ]
        total_prompt_count = len(uid_abs_max_adv)
        kept_prompt_count = len(kept_prompt_uids)
        return kept_indices, total_prompt_count, kept_prompt_count

    @staticmethod
    def _sample_full_prompt_indices(
        batch: DataProto, *, target_prompt_count: int, rollout_repeat: int
    ) -> list[int]:
        if "uid" not in batch.non_tensor_batch:
            raise KeyError("Prompt-level sampling requires non_tensor field 'uid'.")
        if target_prompt_count <= 0:
            raise ValueError(f"target_prompt_count must be positive, got {target_prompt_count}")
        if rollout_repeat <= 0:
            raise ValueError(f"rollout_repeat must be positive, got {rollout_repeat}")

        uid_to_indices: dict[str, list[int]] = {}
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            uid_to_indices.setdefault(str(uid), []).append(idx)

        malformed = {
            uid: len(indices)
            for uid, indices in uid_to_indices.items()
            if len(indices) != rollout_repeat
        }
        if malformed:
            preview = list(malformed.items())[:5]
            raise ValueError(
                "Final CRRL batch sampling requires each prompt to have exactly "
                f"{rollout_repeat} trajectories, but found malformed counts: {preview}"
            )

        prompt_uids = np.array(list(uid_to_indices.keys()), dtype=object)
        if prompt_uids.size < target_prompt_count:
            raise ValueError(
                "Not enough kept prompts to build the final CRRL batch: "
                f"kept={prompt_uids.size}, target={target_prompt_count}"
            )

        selected_positions = np.random.choice(prompt_uids.size, size=target_prompt_count, replace=False)
        selected_indices: list[int] = []
        for pos in selected_positions.tolist():
            selected_indices.extend(uid_to_indices[str(prompt_uids[pos])])

        expected_traj_count = target_prompt_count * rollout_repeat
        if len(selected_indices) != expected_traj_count:
            raise RuntimeError(
                "Prompt-level sampling produced an unexpected number of trajectories: "
                f"selected={len(selected_indices)}, expected={expected_traj_count}"
            )
        return selected_indices

    @staticmethod
    def _runtime_estimator_to_bundle(estimator) -> dict:
        return {
            "bundle_type": "single_trajectory_estimator",
            "bundle_version": 1,
            "config": estimator.config.to_dict(),
            "estimator": estimator.estimator,
            "prompt_hidden_pca": estimator.prompt_hidden_projection,
            "response_hidden_pca": estimator.response_hidden_projection,
        }

    @staticmethod
    def _build_estimator_from_bundle(bundle: dict):
        from recipe.CrossRolloutRL.estimator.single_trajectory_estimator import (
            SingleTrajectoryEstimator,
            SingleTrajectoryEstimatorConfig,
        )

        config_payload = bundle.get("config")
        if config_payload is None:
            raise ValueError("Estimator bundle does not contain 'config'.")
        return SingleTrajectoryEstimator(
            config=SingleTrajectoryEstimatorConfig.from_dict(config_payload),
            estimator=bundle["estimator"],
            prompt_hidden_projection=bundle.get("prompt_hidden_pca"),
            response_hidden_projection=bundle.get(
                "response_hidden_pca",
                bundle.get("think_end_hidden_pca", bundle.get("trajectory_hidden_pca")),
            ),
        )

    def _set_adaptive_estimator_checkpoint_state(
        self,
        *,
        estimator_model_path: str | None,
        estimator_bundle: dict | None,
        retrain_steps: int,
        retrain_count: int,
        train_prompt_hidden_rows: list[np.ndarray],
        train_response_hidden_rows: list[np.ndarray],
        train_response_feature_rows: list[dict[str, float]],
        train_targets: list[float],
        train_step_row_counts: list[int],
    ) -> None:
        self._adaptive_estimator_checkpoint_state = {
            "enabled": True,
            "estimator_model_path": estimator_model_path,
            "retrain_steps": int(retrain_steps),
            "retrain_count": int(retrain_count),
            "train_prompt_hidden_rows": self._normalize_hidden_rows(train_prompt_hidden_rows),
            "train_response_hidden_rows": self._normalize_hidden_rows(train_response_hidden_rows),
            "train_response_feature_rows": self._normalize_feature_rows(train_response_feature_rows),
            "train_targets": self._normalize_targets(train_targets),
            "train_step_row_counts": self._normalize_step_row_counts(train_step_row_counts),
        }
        self._adaptive_estimator_checkpoint_bundle = estimator_bundle

    def _clear_adaptive_estimator_checkpoint_state(self) -> None:
        self._adaptive_estimator_checkpoint_state = None
        self._adaptive_estimator_checkpoint_bundle = None

    @staticmethod
    def _save_estimator_bundle_or_copy(
        *,
        estimator_bundle: dict | None,
        estimator_model_path: str | None,
        snapshot_path: str,
    ) -> tuple[str | None, bool]:
        resolved_model_path = None
        if estimator_model_path:
            resolved_model_path = os.path.abspath(os.path.expanduser(str(estimator_model_path)))

        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        if estimator_bundle is not None:
            try:
                import joblib
            except ImportError as exc:
                raise ImportError("joblib is required to save adaptive estimator snapshots.") from exc
            joblib.dump(estimator_bundle, snapshot_path)
        elif resolved_model_path:
            if os.path.exists(resolved_model_path):
                shutil.copy2(resolved_model_path, snapshot_path)
            else:
                print(
                    "[WARN] Adaptive estimator model snapshot source is missing; "
                    f"keep original path only: {resolved_model_path}"
                )

        if os.path.exists(snapshot_path):
            return snapshot_path, True
        return resolved_model_path, False

    def _save_adaptive_estimator_checkpoint_state(self, checkpoint_dir: str) -> None:
        state = getattr(self, "_adaptive_estimator_checkpoint_state", None)
        if not state or not state.get("enabled", False):
            return

        os.makedirs(checkpoint_dir, exist_ok=True)
        saved_state = dict(state)
        estimator_bundle = getattr(self, "_adaptive_estimator_checkpoint_bundle", None)

        snapshot_path = os.path.join(checkpoint_dir, self._ESTIMATOR_SNAPSHOT_FILENAME)
        saved_model_path, _ = self._save_estimator_bundle_or_copy(
            estimator_bundle=estimator_bundle,
            estimator_model_path=saved_state.get("estimator_model_path"),
            snapshot_path=snapshot_path,
        )
        saved_state["estimator_model_path"] = saved_model_path

        state_path = os.path.join(checkpoint_dir, self._ESTIMATOR_RESUME_STATE_FILENAME)
        torch.save(saved_state, state_path)

        meta_path = os.path.join(checkpoint_dir, self._ESTIMATOR_RESUME_META_FILENAME)
        meta_payload = {
            "schema_version": 1,
            "retrain_steps": int(saved_state.get("retrain_steps", 0)),
            "retrain_count": int(saved_state.get("retrain_count", 0)),
            "buffer_rows": int(len(saved_state.get("train_targets", []))),
            "buffer_steps": int(len(saved_state.get("train_step_row_counts", []))),
            "buffer_step_row_counts": [int(v) for v in saved_state.get("train_step_row_counts", [])],
            "estimator_model_path": saved_state.get("estimator_model_path"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

    def _save_adaptive_estimator_update_snapshot(
        self,
        *,
        output_dir: str,
        estimator_model_path: str | None,
        estimator_bundle: dict | None,
        retrain_steps: int,
        retrain_count: int,
        train_targets: list[float],
        train_step_row_counts: list[int],
    ) -> str | None:
        update_dir = os.path.join(output_dir, f"global_step_{self.global_steps}")
        snapshot_path = os.path.join(update_dir, self._ESTIMATOR_SNAPSHOT_FILENAME)
        saved_model_path, snapshot_saved = self._save_estimator_bundle_or_copy(
            estimator_bundle=estimator_bundle,
            estimator_model_path=estimator_model_path,
            snapshot_path=snapshot_path,
        )

        meta_payload = {
            "schema_version": 1,
            "global_step": int(self.global_steps),
            "retrain_steps": int(retrain_steps),
            "retrain_count": int(retrain_count),
            "buffer_rows": int(len(train_targets)),
            "buffer_steps": int(len(train_step_row_counts)),
            "buffer_step_row_counts": [int(v) for v in train_step_row_counts],
            "estimator_model_path": saved_model_path,
            "snapshot_saved": bool(snapshot_saved),
        }
        meta_path = os.path.join(update_dir, self._ESTIMATOR_UPDATE_META_FILENAME)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        os.makedirs(output_dir, exist_ok=True)
        latest_path = os.path.join(output_dir, self._ESTIMATOR_LATEST_UPDATE_FILENAME)
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(str(self.global_steps))

        return saved_model_path if snapshot_saved else None

    def _resolve_loaded_checkpoint_dir(self) -> str | None:
        if self.global_steps <= 0:
            return None

        if self.config.trainer.resume_mode == "resume_path":
            resume_path = self.config.trainer.resume_from_path
            if isinstance(resume_path, str):
                if not os.path.isabs(resume_path):
                    resume_path = os.path.join(os.getcwd(), resume_path)
                return resume_path

        return self._current_global_step_ckpt_dir()

    def _load_adaptive_estimator_checkpoint_state(self) -> None:
        self._loaded_adaptive_estimator_checkpoint_state = None
        checkpoint_dir = self._resolve_loaded_checkpoint_dir()
        if checkpoint_dir is None:
            return

        state_path = os.path.join(checkpoint_dir, self._ESTIMATOR_RESUME_STATE_FILENAME)
        if not os.path.exists(state_path):
            return

        loaded_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded_state, dict) or not loaded_state.get("enabled", False):
            return
        self._loaded_adaptive_estimator_checkpoint_state = loaded_state
        print(
            "[DEBUG] Loaded adaptive estimator resume state from "
            f"{state_path} (buffer_rows={len(loaded_state.get('train_targets', []))})"
        )

    def _pop_loaded_adaptive_estimator_checkpoint_state(self) -> dict | None:
        state = getattr(self, "_loaded_adaptive_estimator_checkpoint_state", None)
        self._loaded_adaptive_estimator_checkpoint_state = None
        return state

    def _save_checkpoint(self):
        super()._save_checkpoint()
        checkpoint_dir = self._current_global_step_ckpt_dir()
        self._save_adaptive_estimator_checkpoint_state(checkpoint_dir)

    def _load_checkpoint(self):
        super()._load_checkpoint()
        self._load_adaptive_estimator_checkpoint_state()

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        reward_extra_infos_dict = {k: v for k, v in reward_extra_infos_dict.items() if k != "acc"}
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=self._json_default))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _decode_model_prompts(self, batch: DataProto) -> list[str]:
        if "prompts" not in batch.batch.keys():
            raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
            if raw_prompts is None:
                return [""] * len(batch.batch)
            return [self._extract_prompt_text(raw_prompt) for raw_prompt in raw_prompts]

        prompt_ids = batch.batch["prompts"]
        prompt_attention_mask = None
        if "attention_mask" in batch.batch.keys():
            attention_mask = batch.batch["attention_mask"]
            if attention_mask.ndim == 2 and attention_mask.shape[0] == prompt_ids.shape[0]:
                prompt_attention_mask = attention_mask[:, : prompt_ids.shape[1]].bool()

        decoded_prompts = []
        pad_token_id = self.tokenizer.pad_token_id
        for row_idx in range(prompt_ids.shape[0]):
            row = prompt_ids[row_idx].detach().cpu()
            if prompt_attention_mask is not None:
                row = row[prompt_attention_mask[row_idx].detach().cpu()]
            elif pad_token_id is not None:
                row = row[row != pad_token_id]
            decoded_prompts.append(self.tokenizer.decode(row.tolist(), skip_special_tokens=False).strip())
        return decoded_prompts

    def _accumulate_prompt_reward_log_rows(
        self,
        *,
        accumulator: dict[str, dict],
        order: list[str],
        batch: DataProto,
        reward_sums: torch.Tensor,
        round_index: int,
    ) -> None:
        model_prompts = self._decode_model_prompts(batch)
        raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
        uids = batch.non_tensor_batch.get("uid", None)
        rewards = reward_sums.detach().to(torch.float32).cpu().tolist()

        for row_idx, reward in enumerate(rewards):
            uid = str(uids[row_idx]) if uids is not None else str(row_idx)
            if uid not in accumulator:
                order.append(uid)
                accumulator[uid] = {
                    "uid": uid,
                    "prompt": model_prompts[row_idx],
                    "raw_prompt": self._extract_prompt_text(raw_prompts[row_idx]) if raw_prompts is not None else "",
                    "rollout_rewards": [],
                    "round_indices": [],
                }
            accumulator[uid]["rollout_rewards"].append(float(reward))
            accumulator[uid]["round_indices"].append(int(round_index))

    def _dump_prompt_reward_log(
        self,
        *,
        output_dir: str,
        accumulator: dict[str, dict],
        order: list[str],
        group_filter_enabled: bool,
        rollout_repeat: int,
        rounds_total: int,
        final_train_batch_rows: int,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        records = []
        for prompt_index, uid in enumerate(order):
            row = accumulator[uid]
            rewards = [float(v) for v in row["rollout_rewards"]]
            reward_mean = float(np.mean(rewards)) if rewards else 0.0
            records.append(
                {
                    "prompt_index": int(prompt_index),
                    "uid": row["uid"],
                    "prompt": row["prompt"],
                    "raw_prompt": row["raw_prompt"],
                    "reward_mean": reward_mean,
                    "num_rollouts": int(len(rewards)),
                    "rollout_rewards": rewards,
                    "round_indices": sorted(set(int(v) for v in row["round_indices"])),
                }
            )

        payload = {
            "schema_version": 1,
            "global_step": int(self.global_steps),
            "group_filter_enabled": bool(group_filter_enabled),
            "rollout_repeat": int(rollout_repeat),
            "rounds_total": int(rounds_total),
            "final_train_batch_rows": int(final_train_batch_rows),
            "prompt_count": int(len(records)),
            "trajectory_count": int(sum(record["num_rollouts"] for record in records)),
            "records": records,
        }

        filename = os.path.join(output_dir, f"{self.global_steps}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=self._json_default)

        latest_path = os.path.join(output_dir, self._PROMPT_REWARD_LATEST_FILENAME)
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(str(self.global_steps))

        return filename

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Override: Get generation batch with CRRL-specific keys.

        CRRL modification: Includes "raw_prompt" in reward_model_keys.
        """
        reward_model_keys = (
            set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt"}) & batch.non_tensor_batch.keys()
        )

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from recipe.CrossRolloutRL.agent_loop import CRRLAgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = CRRLAgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    @staticmethod
    def _extract_prompt_text(raw_prompt_item) -> str:
        if isinstance(raw_prompt_item, str):
            return raw_prompt_item.strip()
        if isinstance(raw_prompt_item, dict):
            return str(raw_prompt_item.get("content", "")).strip()
        if isinstance(raw_prompt_item, list | tuple) and raw_prompt_item:
            first_turn = raw_prompt_item[0]
            if isinstance(first_turn, dict):
                return str(first_turn.get("content", "")).strip()
            return str(first_turn).strip()
        return str(raw_prompt_item).strip()

    @staticmethod
    def _span_to_token_indices(offsets: list[tuple[int, int]], span: tuple[int, int] | None) -> list[int]:
        if span is None:
            return []
        span_start, span_end = span
        indices = []
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= span_start or tok_start >= span_end:
                continue
            if tok_end > tok_start:
                indices.append(idx)
        return indices

    @staticmethod
    def _mean_at_indices(values: np.ndarray, indices: list[int]) -> float:
        if not indices or values.size == 0:
            return 0.0
        clipped = [idx for idx in indices if 0 <= idx < values.shape[0]]
        if not clipped:
            return 0.0
        return float(values[clipped].mean())

    @staticmethod
    def _entropy_stats(values: np.ndarray, indices: list[int] | None = None) -> dict[str, float]:
        if values.size == 0:
            return {"mean": 0.0, "last": 0.0, "max": 0.0, "min": 0.0}
        if indices is None:
            selected = values
        else:
            clipped = [idx for idx in indices if 0 <= idx < values.shape[0]]
            selected = values[clipped] if clipped else np.asarray([], dtype=values.dtype)
        if selected.size == 0:
            return {"mean": 0.0, "last": 0.0, "max": 0.0, "min": 0.0}
        return {
            "mean": float(selected.mean()),
            "last": float(selected[-1]),
            "max": float(selected.max()),
            "min": float(selected.min()),
        }

    def _find_reasoning_answer_spans(self, text: str) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        reasoning_span = None
        answer_span = None

        think_match = self._THINK_CONTENT_PATTERN.search(text)
        if think_match:
            reasoning_span = (think_match.start(1), think_match.end(1))

        answer_match = self._ANSWER_CONTENT_PATTERN.search(text)
        if answer_match:
            answer_span = (answer_match.start(1), answer_match.end(1))
        elif "</think>" in text:
            start = text.find("</think>") + len("</think>")
            end = len(text)
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                answer_span = (start, end)

        return reasoning_span, answer_span

    def _fallback_segment_indices(
        self, generated_text: str, total_tokens: int
    ) -> tuple[list[int], list[int]]:
        from recipe.CrossRolloutRL.estimator.single_trajectory_estimator_support.feature_builder.features import (
            extract_reasoning_and_answer,
        )

        _, reasoning_text, answer_text = extract_reasoning_and_answer({"generated_text": generated_text})
        reasoning_count = len(self.tokenizer.encode(reasoning_text, add_special_tokens=False)) if reasoning_text else 0
        answer_count = len(self.tokenizer.encode(answer_text, add_special_tokens=False)) if answer_text else 0

        reasoning_count = min(reasoning_count, total_tokens)
        answer_count = min(answer_count, total_tokens)
        reasoning_indices = list(range(reasoning_count))
        answer_indices = list(range(max(0, total_tokens - answer_count), total_tokens))
        return reasoning_indices, answer_indices

    def _compute_rollout_entropy_features(
        self,
        *,
        generated_text: str,
        response_ids: list[int],
        response_entropies: np.ndarray,
    ) -> dict[str, float]:
        reasoning_indices: list[int] = []
        answer_indices: list[int] = []
        use_offset_fallback = False

        try:
            encoded = self.tokenizer(generated_text, add_special_tokens=False, return_offsets_mapping=True)
            token_ids_by_offset = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if token_ids_by_offset and isinstance(token_ids_by_offset[0], list):
                token_ids_by_offset = token_ids_by_offset[0]
            if offsets and isinstance(offsets[0], list):
                offsets = offsets[0]
            offsets = [(int(start), int(end)) for start, end in offsets]

            if len(token_ids_by_offset) != len(response_ids):
                use_offset_fallback = True
            else:
                reasoning_span, answer_span = self._find_reasoning_answer_spans(generated_text)
                reasoning_indices = self._span_to_token_indices(offsets, reasoning_span)
                answer_indices = self._span_to_token_indices(offsets, answer_span)
        except Exception:
            use_offset_fallback = True

        if use_offset_fallback:
            reasoning_indices, answer_indices = self._fallback_segment_indices(generated_text, len(response_ids))

        output_stats = self._entropy_stats(response_entropies)
        reasoning_stats = self._entropy_stats(response_entropies, reasoning_indices)
        answer_stats = self._entropy_stats(response_entropies, answer_indices)
        return {
            "output_mean_token_entropy": output_stats["mean"],
            "output_last_token_entropy": output_stats["last"],
            "output_max_token_entropy": output_stats["max"],
            "output_min_token_entropy": output_stats["min"],
            "reasoning_mean_token_entropy": reasoning_stats["mean"],
            "reasoning_last_token_entropy": reasoning_stats["last"],
            "reasoning_max_token_entropy": reasoning_stats["max"],
            "reasoning_min_token_entropy": reasoning_stats["min"],
            "answer_mean_token_entropy": answer_stats["mean"],
            "answer_last_token_entropy": answer_stats["last"],
            "answer_max_token_entropy": answer_stats["max"],
            "answer_min_token_entropy": answer_stats["min"],
        }

    def _compute_estimator_cross_rollout_advantages(
        self,
        *,
        batch: DataProto,
        reward_sums: torch.Tensor,
        token_entropys: torch.Tensor,
        estimator,
        feature_builder,
        pair_size: int,
        target_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list]]:
        if pair_size != 2:
            raise ValueError(f"Estimator cross-rollout advantage currently requires pair_size=2, got {pair_size}.")
        if target_mode not in {"pair_average", "other_rollout_correctness"}:
            raise ValueError(
                "Estimator target_mode must be one of {'pair_average', 'other_rollout_correctness'}, "
                f"got {target_mode!r}."
            )

        prompt_hidden_key = "estimator_prompt_hidden"
        response_hidden_key = "estimator_response_hidden"

        if prompt_hidden_key not in batch.batch.keys() or response_hidden_key not in batch.batch.keys():
            raise KeyError(
                "Estimator hidden features are missing in batch. "
                f"Expected '{prompt_hidden_key}' and '{response_hidden_key}'."
            )

        if "uid" not in batch.non_tensor_batch:
            raise KeyError("Cross-rollout estimator advantage requires non_tensor field 'uid'.")

        batch_size = len(batch.batch)
        value_predictions = torch.zeros(batch_size, dtype=torch.float32)
        prompt_hidden_rows: list[np.ndarray] = []
        response_hidden_rows: list[np.ndarray] = []
        response_feature_rows: list[dict[str, float]] = []
        response_masks = batch.batch["response_mask"]
        response_tokens = batch.batch["responses"]
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")

        for idx in range(batch_size):
            valid_mask = response_masks[idx].bool().detach().cpu()
            valid_response_ids = response_tokens[idx].detach().cpu()[valid_mask].tolist()
            generated_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            entropy_values = token_entropys[idx].detach().cpu()[valid_mask].numpy()
            rollout_features = self._compute_rollout_entropy_features(
                generated_text=generated_text,
                response_ids=valid_response_ids,
                response_entropies=entropy_values,
            )

            prompt_hidden_raw = batch.batch[prompt_hidden_key][idx].detach().cpu().numpy()
            response_hidden_raw = batch.batch[response_hidden_key][idx].detach().cpu().numpy()

            reasoning_content = ""
            answer_content = ""
            if raw_prompts is not None:
                prompt_text = self._extract_prompt_text(raw_prompts[idx])
                if prompt_text and generated_text.startswith(prompt_text):
                    generated_text = generated_text[len(prompt_text) :]
                generated_text = generated_text.strip()

            prompt_hidden, response_hidden, response_features = feature_builder.build_inputs(
                prompt_hidden_layers=prompt_hidden_raw,
                response_hidden_layers=response_hidden_raw,
                generated_text=generated_text,
                response_ids=valid_response_ids,
                tokenizer=self.tokenizer,
                reasoning_content=reasoning_content,
                answer_content=answer_content,
                rollout_features=rollout_features,
            )
            prompt_hidden_rows.append(np.asarray(prompt_hidden, dtype=np.float32).reshape(-1))
            response_hidden_rows.append(np.asarray(response_hidden, dtype=np.float32).reshape(-1))
            response_feature_rows.append({k: float(v) for k, v in response_features.items()})
            value_predictions[idx] = float(
                estimator.predict_value(
                    prompt_hidden=prompt_hidden,
                    response_hidden=response_hidden,
                    response_features=response_features,
                )
            )

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            uid_to_indices[str(uid)].append(idx)

        cross_baselines = torch.zeros_like(value_predictions)
        target_tensor = torch.zeros(batch_size, dtype=torch.float32)
        for uid, indices in uid_to_indices.items():
            if len(indices) != pair_size:
                raise ValueError(
                    "Cross-rollout estimator advantage requires each uid to appear exactly twice. "
                    f"uid={uid}, count={len(indices)}."
                )
            first_idx, second_idx = indices
            # Each prediction estimates the sibling rollout's reward/target, so it is used as
            # the sibling row's baseline when computing advantages.
            cross_baselines[first_idx] = value_predictions[second_idx]
            cross_baselines[second_idx] = value_predictions[first_idx]
            if target_mode == "pair_average":
                pair_avg_target = float(reward_sums[indices].to(torch.float32).mean().detach().cpu().item())
                target_tensor[first_idx] = pair_avg_target
                target_tensor[second_idx] = pair_avg_target
            else:
                target_tensor[first_idx] = float(reward_sums[second_idx].to(torch.float32).detach().cpu().item())
                target_tensor[second_idx] = float(reward_sums[first_idx].to(torch.float32).detach().cpu().item())

        raw_advantages = reward_sums.to(torch.float32) - cross_baselines.to(torch.float32)
        training_rows = {
            "prompt_hidden_rows": prompt_hidden_rows,
            "response_hidden_rows": response_hidden_rows,
            "response_feature_rows": response_feature_rows,
            "targets": target_tensor.tolist(),
        }
        return raw_advantages, cross_baselines, value_predictions, training_rows

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                logger.close()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        crrl_mode = "non_adaptive_estimator"
        crrl_missing_prompt = "error"
        crrl_default_p_hat = 0.5
        crrl_weighted_sampling = True
        full_prompts = []
        prompt2baseline = {}
        prompt2sampled_number = defaultdict(int)
        estimator_enabled = False
        estimator = None
        estimator_feature_builder = None
        prompt_reward_log_dir = None
        estimator_feature_builder_config = None
        estimator_fit_config = None
        estimator_fit_single_fn = None
        estimator_current_model_path = None
        estimator_current_bundle = None
        estimator_update_output_dir = None
        estimator_pair_size = 2
        estimator_capture_spec = None
        estimator_warmup_steps = 4
        estimator_retrain_steps = 0
        estimator_retrain_count = 0
        estimator_buffer_max_steps = 4
        estimator_train_prompt_hidden_rows: list[np.ndarray] = []
        estimator_train_response_hidden_rows: list[np.ndarray] = []
        estimator_train_response_feature_rows: list[dict[str, float]] = []
        estimator_train_targets: list[float] = []
        estimator_train_step_row_counts: list[int] = []
        crrl_group_filter_cfg = self.config.trainer.crrl.get("group_filter", {})
        crrl_group_filter_enabled = self._config_bool(
            crrl_group_filter_cfg.get("enable", True), name="trainer.crrl.group_filter.enable"
        )

        if self.config.trainer.crrl.enable:
            prompt_reward_log_dir = self._resolve_prompt_reward_log_dir(
                self.config.trainer.crrl.get("prompt_reward_log_dir", None)
            )
            crrl_mode = str(self.config.trainer.crrl.get("mode", "non_adaptive_estimator")).lower()
            if crrl_mode not in {"non_adaptive_estimator", "adaptive_estimator"}:
                raise ValueError(
                    f"Unknown trainer.crrl.mode: {crrl_mode}. "
                    "Expected one of non_adaptive_estimator, adaptive_estimator."
                )

            crrl_missing_prompt = str(self.config.trainer.crrl.get("missing_prompt", "error")).lower()
            if crrl_missing_prompt not in {"error", "default"}:
                raise ValueError(
                    f"Unknown trainer.crrl.missing_prompt: {crrl_missing_prompt}. Expected one of error, default."
                )

            crrl_default_p_hat = float(self.config.trainer.crrl.get("default_p_hat", 0.5))
            if not (0.0 <= crrl_default_p_hat <= 1.0):
                raise ValueError(f"trainer.crrl.default_p_hat must be in [0, 1], got: {crrl_default_p_hat}")

            crrl_weighted_sampling = self._config_bool(
                self.config.trainer.crrl.get("weighted_sampling", True), name="trainer.crrl.weighted_sampling"
            )
            estimator_enabled = crrl_mode == "adaptive_estimator"

            if crrl_mode == "non_adaptive_estimator":
                baseline_values_path = self.config.trainer.crrl.get("baseline_values", None)
                if not baseline_values_path:
                    raise ValueError(
                        "trainer.crrl.baseline_values must be provided when "
                        "trainer.crrl.mode='non_adaptive_estimator'."
                    )

                prompt2baseline = {}
                try:
                    # Supports classic JSON object format: {"prompt": p_hat, ...}
                    loaded_baseline = json.load(open(baseline_values_path))
                    if isinstance(loaded_baseline, dict):
                        for prompt, baseline in loaded_baseline.items():
                            baseline = float(baseline)
                            if not (0.0 <= baseline <= 1.0):
                                raise ValueError(
                                    f"Baseline p_hat must be in [0, 1] for prompt='{prompt[:80]}...', got: {baseline}"
                                )
                            prompt2baseline[prompt.strip()] = baseline
                    else:
                        raise TypeError(
                            f"Expected JSON object for baseline values, got {type(loaded_baseline)}; "
                            "falling back to JSONL parser."
                        )
                except Exception:
                    # Supports JSONL format:
                    # {"prompt": "...", "p_hat": 0.73}
                    # {"prompt": "...", "predicted_value": 0.73}
                    with open(baseline_values_path, encoding="utf-8") as f:
                        for line_idx, line in enumerate(f, start=1):
                            line = line.strip()
                            if not line:
                                continue
                            item = json.loads(line)
                            if not isinstance(item, dict):
                                raise TypeError(
                                    "JSONL baseline line must be an object. "
                                    f"Got {type(item)} at line {line_idx} in {baseline_values_path}"
                                )
                            if "prompt" not in item:
                                raise KeyError(
                                    f"JSONL baseline line must contain 'prompt'. "
                                    f"Missing at line {line_idx} in {baseline_values_path}"
                                )

                            baseline = item.get("p_hat", item.get("predicted_value", item.get("baseline")))
                            if baseline is None:
                                raise KeyError(
                                    "JSONL baseline line must contain one of 'p_hat', 'predicted_value', or 'baseline'. "
                                    f"Missing at line {line_idx} in {baseline_values_path}"
                                )
                            baseline = float(baseline)
                            if not (0.0 <= baseline <= 1.0):
                                raise ValueError(
                                    f"Baseline p_hat must be in [0, 1] for prompt='{item['prompt'][:80]}...', got: {baseline}"
                                )
                            prompt2baseline[item["prompt"].strip()] = baseline

                            # Allow baseline matching when training prompt_key=source_prompt.
                            source_prompt = item.get("source_prompt")
                            source_prompt_content = None
                            if isinstance(source_prompt, list | tuple) and source_prompt:
                                first_turn = source_prompt[0]
                                if isinstance(first_turn, dict):
                                    source_prompt_content = first_turn.get("content")
                            elif isinstance(source_prompt, dict):
                                source_prompt_content = source_prompt.get("content")

                            if isinstance(source_prompt_content, str):
                                prompt2baseline[source_prompt_content.strip()] = baseline

                full_prompts = list(prompt2baseline.keys())
                print(
                    f"[DEBUG] Loaded {len(prompt2baseline)} baseline prompts from "
                    f"trainer.crrl.baseline_values={baseline_values_path}"
                )
            else:
                if crrl_weighted_sampling:
                    raise ValueError(
                        "trainer.crrl.weighted_sampling must be False when "
                        "trainer.crrl.mode='adaptive_estimator'."
                    )
                crrl_estimator_cfg = self.config.trainer.crrl.get("estimator", None)
                if crrl_estimator_cfg is None:
                    crrl_estimator_cfg = {}

                if self.config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
                    raise NotImplementedError(
                        "trainer.crrl.estimator currently supports actor strategy in {'fsdp', 'fsdp2'} only."
                    )
                from recipe.CrossRolloutRL.estimator.single_trajectory_estimator import (
                    FeatureBuilderConfig,
                    SingleTrajectoryEstimatorFitConfig,
                    SingleTrajectoryFeatureBuilder,
                    fit_single_trajectory_estimator,
                    load_single_trajectory_estimator,
                )

                estimator_model_path = crrl_estimator_cfg.get("model_path", None)
                if not estimator_model_path:
                    raise ValueError(
                        "trainer.crrl.estimator.model_path must be set when trainer.crrl.mode='adaptive_estimator'."
                    )
                estimator = load_single_trajectory_estimator(estimator_model_path)
                estimator_current_model_path = estimator_model_path
                estimator_current_bundle = self._runtime_estimator_to_bundle(estimator)

                feature_builder_config_path = crrl_estimator_cfg.get(
                    "feature_builder_config_path",
                    "recipe/CrossRolloutRL/estimator/single_trajectory_estimator_support/default_feature_builder_config.json",
                )
                with open(feature_builder_config_path, encoding="utf-8") as f:
                    feature_builder_payload = json.load(f)
                feature_builder_config = FeatureBuilderConfig.from_dict(feature_builder_payload)
                estimator_feature_builder = SingleTrajectoryFeatureBuilder(feature_builder_config)
                estimator_feature_builder_config = feature_builder_config
                estimator_pair_size = int(crrl_estimator_cfg.get("pair_size", 2))
                estimator_warmup_steps = int(crrl_estimator_cfg.get("warmup_steps", 4))
                if estimator_warmup_steps <= 0:
                    raise ValueError(
                        "trainer.crrl.estimator.warmup_steps must be a positive integer, "
                        f"got {estimator_warmup_steps}."
                    )
                estimator_buffer_max_steps = int(crrl_estimator_cfg.get("buffer_max_steps", 4))
                if estimator_buffer_max_steps <= 0:
                    raise ValueError(
                        "trainer.crrl.estimator.buffer_max_steps must be a positive integer, "
                        f"got {estimator_buffer_max_steps}."
                    )

                fit_config_path = crrl_estimator_cfg.get(
                    "fit_config_path",
                    "recipe/CrossRolloutRL/estimator/single_trajectory_estimator_support/default_estimator_fit_config.json",
                )
                with open(fit_config_path, encoding="utf-8") as f:
                    fit_payload = json.load(f)
                estimator_fit_config = SingleTrajectoryEstimatorFitConfig(**fit_payload)
                if estimator_fit_config.target_mode not in {"pair_average", "other_rollout_correctness"}:
                    raise ValueError(
                        "trainer.crrl.estimator.fit_config target_mode must be one of "
                        "{'pair_average', 'other_rollout_correctness'}, "
                        f"got {estimator_fit_config.target_mode!r}."
                    )
                estimator_fit_single_fn = fit_single_trajectory_estimator

                estimator_update_output_dir = self._resolve_estimator_update_output_dir(
                    crrl_estimator_cfg.get("online_output_dir", None)
                )

                if feature_builder_config.prompt_hidden.layer_index != feature_builder_config.response_hidden.layer_index:
                    raise ValueError(
                        "trainer.crrl.estimator currently requires prompt/response hidden layer_index to match."
                    )
                if feature_builder_config.prompt_hidden.pooling.type != "last_n_mean":
                    raise ValueError("trainer.crrl.estimator expects prompt_hidden.pooling.type='last_n_mean'.")
                if feature_builder_config.response_hidden.pooling.type != "last_n_mean":
                    raise ValueError("trainer.crrl.estimator expects response_hidden.pooling.type='last_n_mean'.")

                think_end_token_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
                estimator_capture_spec = {
                    "layer_index": int(feature_builder_config.prompt_hidden.layer_index),
                    "prompt_pool_n": int(feature_builder_config.prompt_hidden.pooling.n),
                    "response_pool_n": int(feature_builder_config.response_hidden.pooling.n),
                    "think_end_token_ids": think_end_token_ids,
                }
                rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                if rollout_n != estimator_pair_size:
                    raise ValueError(
                        "Cross-rollout estimator requires actor_rollout_ref.rollout.n == "
                        f"trainer.crrl.estimator.pair_size. Got n={rollout_n}, pair_size={estimator_pair_size}."
                    )

                full_prompts = []
                print(
                    "[DEBUG] Enabled CRRL estimator baseline: "
                    f"model={estimator_model_path}, pair_size={estimator_pair_size}, "
                    f"hidden_source=pi_theta, warmup_steps={estimator_warmup_steps}, "
                    f"fifo_max_steps={estimator_buffer_max_steps}, "
                    "update_every=1step_after_warmup, "
                    f"update_snapshot_dir={estimator_update_output_dir}, "
                    "persistence=checkpoint_and_update_snapshots"
                )

                loaded_estimator_state = self._pop_loaded_adaptive_estimator_checkpoint_state()
                if loaded_estimator_state is not None:
                    loaded_model_path = loaded_estimator_state.get("estimator_model_path")
                    resolved_loaded_model_path = None
                    if loaded_model_path:
                        resolved_loaded_model_path = os.path.abspath(os.path.expanduser(str(loaded_model_path)))

                    if resolved_loaded_model_path and os.path.exists(resolved_loaded_model_path):
                        estimator = load_single_trajectory_estimator(resolved_loaded_model_path)
                        estimator_current_model_path = resolved_loaded_model_path
                        estimator_current_bundle = self._runtime_estimator_to_bundle(estimator)
                    elif loaded_model_path:
                        print(
                            "[WARN] Resume estimator model path does not exist. "
                            f"Fallback to configured model_path={estimator_model_path}: {resolved_loaded_model_path}"
                        )

                    estimator_retrain_steps = int(loaded_estimator_state.get("retrain_steps", 0))
                    estimator_retrain_count = int(loaded_estimator_state.get("retrain_count", 0))
                    estimator_train_prompt_hidden_rows = self._normalize_hidden_rows(
                        loaded_estimator_state.get("train_prompt_hidden_rows")
                    )
                    estimator_train_response_hidden_rows = self._normalize_hidden_rows(
                        loaded_estimator_state.get("train_response_hidden_rows")
                    )
                    estimator_train_response_feature_rows = self._normalize_feature_rows(
                        loaded_estimator_state.get("train_response_feature_rows")
                    )
                    estimator_train_targets = self._normalize_targets(loaded_estimator_state.get("train_targets"))
                    estimator_train_step_row_counts = self._normalize_step_row_counts(
                        loaded_estimator_state.get("train_step_row_counts")
                    )

                    prompt_rows_len = len(estimator_train_prompt_hidden_rows)
                    response_rows_len = len(estimator_train_response_hidden_rows)
                    feature_rows_len = len(estimator_train_response_feature_rows)
                    target_rows_len = len(estimator_train_targets)
                    if len({prompt_rows_len, response_rows_len, feature_rows_len, target_rows_len}) != 1:
                        raise ValueError(
                            "Loaded adaptive estimator resume buffer has inconsistent row counts: "
                            f"prompt={prompt_rows_len}, response={response_rows_len}, "
                            f"feature={feature_rows_len}, targets={target_rows_len}"
                        )
                    if not estimator_train_step_row_counts and target_rows_len > 0:
                        estimator_train_step_row_counts = [target_rows_len]
                    if sum(estimator_train_step_row_counts) != target_rows_len:
                        print(
                            "[WARN] Adaptive estimator resume step row counts do not match row buffer size. "
                            "Treating restored rows as one legacy step."
                        )
                        estimator_train_step_row_counts = [target_rows_len] if target_rows_len > 0 else []
                    restored_trimmed_steps = 0
                    restored_trimmed_rows = 0
                    while len(estimator_train_step_row_counts) > estimator_buffer_max_steps:
                        rows_to_trim = estimator_train_step_row_counts.pop(0)
                        restored_trimmed_rows += rows_to_trim
                        restored_trimmed_steps += 1
                        estimator_train_prompt_hidden_rows = estimator_train_prompt_hidden_rows[rows_to_trim:]
                        estimator_train_response_hidden_rows = estimator_train_response_hidden_rows[rows_to_trim:]
                        estimator_train_response_feature_rows = estimator_train_response_feature_rows[rows_to_trim:]
                        estimator_train_targets = estimator_train_targets[rows_to_trim:]
                        target_rows_len = len(estimator_train_targets)
                    if restored_trimmed_steps > 0:
                        print(
                            "[DEBUG] Trimmed restored adaptive estimator FIFO buffer to "
                            f"{len(estimator_train_step_row_counts)} steps / {target_rows_len} rows "
                            f"(max_steps={estimator_buffer_max_steps}, "
                            f"trimmed_steps={restored_trimmed_steps}, "
                            f"trimmed_rows={restored_trimmed_rows})."
                        )
                    print(
                        "[DEBUG] Restored adaptive estimator state from checkpoint: "
                        f"retrain_steps={estimator_retrain_steps}, retrain_count={estimator_retrain_count}, "
                        f"buffer_steps={len(estimator_train_step_row_counts)}, "
                        f"buffer_rows={target_rows_len}, model={estimator_current_model_path}"
                    )
            if not estimator_enabled:
                loaded_estimator_state = self._pop_loaded_adaptive_estimator_checkpoint_state()
                if loaded_estimator_state is not None:
                    print(
                        "[WARN] Adaptive estimator resume state exists but current mode is "
                        f"{crrl_mode}. Ignoring estimator resume payload."
                    )
        else:
            loaded_estimator_state = self._pop_loaded_adaptive_estimator_checkpoint_state()
            if loaded_estimator_state is not None:
                print(
                    "[WARN] Adaptive estimator resume state exists but trainer.crrl.enable is False. "
                    "Ignoring estimator resume payload."
                )

        target_prompt_batch_size = int(self.config.data.train_batch_size)
        rollout_repeat = int(self.config.actor_rollout_ref.rollout.n)
        rollout_agent_workers = (
            int(self.config.actor_rollout_ref.rollout.agent.num_workers)
            if self.config.actor_rollout_ref.rollout.mode == "async"
            else 1
        )
        if rollout_agent_workers <= 0:
            raise ValueError(
                "actor_rollout_ref.rollout.agent.num_workers must be positive, "
                f"got {rollout_agent_workers}"
            )
        prompt_chunk_divisor = rollout_agent_workers // math.gcd(rollout_agent_workers, rollout_repeat)
        default_br_size = int(self.config.data.get("default_br_size", 768))
        if default_br_size <= 0:
            raise ValueError(f"data.default_br_size must be a positive integer, got {default_br_size}")
        adaptive_beta = float(self.config.data.get("beta", 1.25))
        if adaptive_beta <= 0:
            raise ValueError(f"data.beta must be positive, got {adaptive_beta}")
        advantage_zero_eps = float(self.config.data.get("advantage_zero_eps", 1e-8))
        if advantage_zero_eps < 0:
            raise ValueError(f"data.advantage_zero_eps must be >= 0, got {advantage_zero_eps}")

        pre_filter_buffer = None
        final_accumulation = None
        final_accumulation_estimator_rows = None
        estimator_fit_accumulation_rows = self._new_estimator_row_buffer()
        estimator_online_value_predictions: list[float] = []
        estimator_online_targets: list[float] = []
        estimator_online_raw_advantages: list[float] = []
        estimator_online_rewards: list[float] = []
        estimator_online_pairwise_sign_matches: list[float] = []
        estimator_online_clip_min_count = 0
        estimator_online_clip_max_count = 0
        estimator_online_clip_min_reward_match_count = 0
        estimator_online_clip_max_reward_match_count = 0
        target_accumulation_size = default_br_size if crrl_group_filter_enabled else target_prompt_batch_size
        seen_prompt_count = 0
        zero_adv_prompt_count = 0
        group_filter_round_count = 0
        group_filter_generated_prompts_total = 0
        group_filter_round_generated_prompt_counts: list[int] = []
        group_filter_reward_sum = 0.0
        group_filter_reward_count = 0
        group_filter_p_hat_sum = 0.0
        group_filter_p_hat_count = 0
        prompt_reward_log_accumulator: dict[str, dict] = {}
        prompt_reward_log_order: list[str] = []
        # Group filtering can span multiple rollout rounds before one logged update.
        timing_raw = defaultdict(float)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.config.trainer.crrl.enable and crrl_weighted_sampling:
                    EXPLORATION_EPSILON = 0.05
                    prompt2phat = prompt2baseline

                    prompt2weight = {
                        k: ((prompt2phat[k] * (1.0 - prompt2phat[k])) ** 0.5) + EXPLORATION_EPSILON
                        for k in full_prompts
                    }
                    default_weight = ((crrl_default_p_hat * (1.0 - crrl_default_p_hat)) ** 0.5) + EXPLORATION_EPSILON

                    items = []
                    weights = []
                    for i, p in enumerate(batch_dict["raw_prompt"]):
                        p_str = p[0]["content"].strip()
                        if p_str in prompt2weight:
                            w = float(prompt2weight[p_str])
                        elif crrl_missing_prompt == "default":
                            w = float(default_weight)
                        else:
                            raise KeyError(
                                "Prompt missing in CRRL baseline map while weighted_sampling=True. "
                                f"prompt='{p_str[:160]}...'"
                            )
                        items.append(i)
                        weights.append(w)

                    M = len(items)
                    if M > 0:
                        weights_np = np.asarray(weights, dtype=np.float64)
                        wsum = float(weights_np.sum())

                        if wsum > 0.0:
                            probs = weights_np / wsum
                        else:
                            probs = np.full(M, 1.0 / M, dtype=np.float64)

                        probs = probs / probs.sum()

                        target_bs = int(self.config.data.train_batch_size)
                        replace = target_bs > M

                        selected_pos = np.random.choice(M, size=target_bs, replace=replace, p=probs)
                        keep_idx = [items[j] for j in selected_pos.tolist()]

                        if keep_idx:
                            sampled_batch_dict = {}
                            for k, v in batch_dict.items():
                                try:
                                    sampled_batch_dict[k] = v[keep_idx]
                                    continue
                                except Exception:
                                    pass

                                if isinstance(v, list | tuple):
                                    sampled_batch_dict[k] = type(v)(v[i] for i in keep_idx)
                                else:
                                    sampled_batch_dict[k] = v

                            batch_dict = sampled_batch_dict
                            print(f"[DEBUG] Final size of keep_idx: {len(keep_idx)}")

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.non_tensor_batch["uid"] = self._build_stable_uids_from_input_ids(new_batch.batch["input_ids"])

                if pre_filter_buffer is None:
                    pre_filter_buffer = new_batch
                else:
                    if crrl_group_filter_enabled:
                        existing_uids = set(pre_filter_buffer.non_tensor_batch["uid"])
                        unique_indices = [
                            i for i, uid in enumerate(new_batch.non_tensor_batch["uid"])
                            if uid not in existing_uids
                        ]
                        if unique_indices:
                            pre_filter_buffer = DataProto.concat([pre_filter_buffer, new_batch[unique_indices]])
                    else:
                        pre_filter_buffer = DataProto.concat([pre_filter_buffer, new_batch])

                if crrl_group_filter_enabled:
                    effective_target_accumulation_size = (
                        (target_accumulation_size + prompt_chunk_divisor - 1) // prompt_chunk_divisor
                    ) * prompt_chunk_divisor
                else:
                    effective_target_accumulation_size = target_prompt_batch_size
                current_prompt_pool_size = (
                    len(set(pre_filter_buffer.non_tensor_batch["uid"]))
                    if crrl_group_filter_enabled
                    else len(pre_filter_buffer.batch)
                )
                if current_prompt_pool_size < effective_target_accumulation_size:
                    continue

                batch = pre_filter_buffer[:effective_target_accumulation_size]
                remaining_pre_filter_buffer = pre_filter_buffer[effective_target_accumulation_size:]
                pre_filter_buffer = (
                    remaining_pre_filter_buffer if len(remaining_pre_filter_buffer) > 0 else None
                )
                current_round_prompt_count = len(set(batch.non_tensor_batch["uid"]))
                group_filter_round_count += 1
                group_filter_generated_prompts_total += current_round_prompt_count
                group_filter_round_generated_prompt_counts.append(current_round_prompt_count)

                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        for timing_name, timing_value in gen_batch_output.meta_info["timing"].items():
                            timing_raw[timing_name] += timing_value
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    micro_prompts = batch.non_tensor_batch.get("raw_prompt", None)
                    micro_prompts = [_[0]["content"].strip() for _ in micro_prompts]
                    crrl_metrics = {}
                    r = None

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        if estimator_enabled and estimator_capture_spec is not None:
                            batch.meta_info["estimator_hidden_capture"] = dict(estimator_capture_spec)
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        token_entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(
                            loss_mat=token_entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        batch.meta_info.pop("estimator_hidden_capture", None)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        raw_advantages_for_filter = None
                        estimator_training_rows = None
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        if self.config.trainer.crrl.enable:
                            r = reward_tensor.sum(dim=-1)
                            group_filter_reward_sum += float(r.to(torch.float32).sum().detach().cpu().item())
                            group_filter_reward_count += int(r.numel())
                            self._accumulate_prompt_reward_log_rows(
                                accumulator=prompt_reward_log_accumulator,
                                order=prompt_reward_log_order,
                                batch=batch,
                                reward_sums=r,
                                round_index=group_filter_round_count,
                            )

                            if estimator_enabled:
                                raw_advantages, p_hats, value_predictions, estimator_training_rows = (
                                    self._compute_estimator_cross_rollout_advantages(
                                        batch=batch,
                                        reward_sums=r,
                                        token_entropys=token_entropys,
                                        estimator=estimator,
                                        feature_builder=estimator_feature_builder,
                                        pair_size=estimator_pair_size,
                                        target_mode=estimator_fit_config.target_mode,
                                    )
                                )
                                p_hats = p_hats.to(r)
                                value_predictions = value_predictions.to(r)
                                estimator_targets = torch.tensor(
                                    estimator_training_rows["targets"],
                                    dtype=torch.float32,
                                    device=value_predictions.device,
                                )
                                self._extend_estimator_row_buffer(
                                    estimator_fit_accumulation_rows, estimator_training_rows
                                )

                                # Fit the estimator on every rollout generated for this actor update,
                                # including rollouts later removed by group filtering.
                                pred_values = value_predictions.to(torch.float32)
                                target_errors = pred_values - estimator_targets
                                estimator_online_value_predictions.extend(
                                    pred_values.detach().cpu().numpy().astype(np.float32, copy=False).tolist()
                                )
                                estimator_online_targets.extend(
                                    estimator_targets.detach().cpu().numpy().astype(np.float32, copy=False).tolist()
                                )
                                estimator_online_raw_advantages.extend(
                                    raw_advantages.to(torch.float32)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32, copy=False)
                                    .tolist()
                                )
                                estimator_online_rewards.extend(
                                    r.to(torch.float32).detach().cpu().numpy().astype(np.float32, copy=False).tolist()
                                )
                                crrl_metrics["crrl/adaptive_estimator/online_target_mae"] = float(
                                    target_errors.abs().mean().detach().cpu().item()
                                )
                                crrl_metrics["crrl/adaptive_estimator/online_target_rmse"] = float(
                                    torch.sqrt(torch.mean(target_errors.square())).detach().cpu().item()
                                )
                                crrl_metrics["crrl/adaptive_estimator/online_target_bias"] = float(
                                    target_errors.mean().detach().cpu().item()
                                )

                                pred_values_np = pred_values.detach().cpu().numpy().astype(np.float32, copy=False)
                                estimator_targets_np = (
                                    estimator_targets.detach().cpu().numpy().astype(np.float32, copy=False)
                                )
                                pred_std = float(pred_values_np.std())
                                target_std = float(estimator_targets_np.std())
                                if pred_values_np.size > 1 and pred_std > 1e-8 and target_std > 1e-8:
                                    online_target_pearson = float(
                                        np.corrcoef(pred_values_np, estimator_targets_np)[0, 1]
                                    )
                                else:
                                    online_target_pearson = 0.0
                                if not np.isfinite(online_target_pearson):
                                    online_target_pearson = 0.0
                                crrl_metrics["crrl/adaptive_estimator/online_target_pearson"] = online_target_pearson

                                uid_to_indices_for_metric: dict[str, list[int]] = defaultdict(list)
                                for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
                                    uid_to_indices_for_metric[str(uid)].append(idx)

                                pairwise_sign_matches: list[float] = []
                                for _, indices in uid_to_indices_for_metric.items():
                                    if len(indices) != estimator_pair_size:
                                        continue
                                    first_idx, second_idx = indices
                                    if estimator_fit_config.target_mode == "other_rollout_correctness":
                                        pred_diff_tensor = value_predictions[second_idx] - value_predictions[first_idx]
                                    else:
                                        pred_diff_tensor = value_predictions[first_idx] - value_predictions[second_idx]
                                    pred_diff = float(pred_diff_tensor.detach().cpu().item())
                                    reward_diff = float((r[first_idx] - r[second_idx]).detach().cpu().item())
                                    pred_sign = np.sign(pred_diff)
                                    reward_sign = np.sign(reward_diff)
                                    pairwise_sign_matches.append(1.0 if pred_sign == reward_sign else 0.0)
                                if pairwise_sign_matches:
                                    estimator_online_pairwise_sign_matches.extend(pairwise_sign_matches)
                                    crrl_metrics["crrl/adaptive_estimator/pairwise_sign_acc"] = float(
                                        np.mean(pairwise_sign_matches)
                                    )
                                else:
                                    crrl_metrics["crrl/adaptive_estimator/pairwise_sign_acc"] = 0.0

                                reward_var = float(r.to(torch.float32).var(unbiased=False).detach().cpu().item())
                                adv_var = float(
                                    raw_advantages.to(torch.float32).var(unbiased=False).detach().cpu().item()
                                )
                                if reward_var > 1e-12:
                                    crrl_metrics["crrl/adaptive_estimator/adv_var_reduction_ratio"] = (
                                        adv_var / reward_var
                                    )
                                else:
                                    crrl_metrics["crrl/adaptive_estimator/adv_var_reduction_ratio"] = 0.0

                                pred_values_np = value_predictions.detach().cpu().numpy().astype(np.float32, copy=False)
                                clip_min = float(estimator.config.model.clip_min)
                                clip_max = float(estimator.config.model.clip_max)
                                crrl_metrics["crrl/adaptive_estimator/pred_clip_frac_min"] = float(
                                    np.mean(pred_values_np <= (clip_min + 1e-6))
                                )
                                crrl_metrics["crrl/adaptive_estimator/pred_clip_frac_max"] = float(
                                    np.mean(pred_values_np >= (clip_max - 1e-6))
                                )

                                r_np = r.detach().cpu().numpy().astype(np.float32, copy=False)
                                mask_at_min = pred_values_np <= (clip_min + 1e-6)
                                mask_at_max = pred_values_np >= (clip_max - 1e-6)
                                estimator_online_clip_min_count += int(mask_at_min.sum())
                                estimator_online_clip_max_count += int(mask_at_max.sum())
                                if mask_at_min.sum() > 0:
                                    estimator_online_clip_min_reward_match_count += int(
                                        np.sum(r_np[mask_at_min] <= (clip_min + 1e-6))
                                    )
                                    crrl_metrics["crrl/adaptive_estimator/pred_clip_min_reward_match_frac"] = float(
                                        np.mean(r_np[mask_at_min] <= (clip_min + 1e-6))
                                    )
                                if mask_at_max.sum() > 0:
                                    estimator_online_clip_max_reward_match_count += int(
                                        np.sum(r_np[mask_at_max] >= (clip_max - 1e-6))
                                    )
                                    crrl_metrics["crrl/adaptive_estimator/pred_clip_max_reward_match_frac"] = float(
                                        np.mean(r_np[mask_at_max] >= (clip_max - 1e-6))
                                    )

                            else:
                                missing_baseline_count = 0
                                baseline_values = []
                                for prompt in micro_prompts:
                                    if prompt in prompt2baseline:
                                        baseline_values.append(prompt2baseline[prompt])
                                    elif crrl_missing_prompt == "default":
                                        baseline_values.append(crrl_default_p_hat)
                                        missing_baseline_count += 1
                                    else:
                                        raise KeyError(
                                            "Prompt missing in trainer.crrl.baseline_values while "
                                            "mode='non_adaptive_estimator'. "
                                            f"prompt='{prompt[:160]}...'"
                                        )
                                p_hats = torch.tensor(baseline_values, dtype=torch.float).to(r)
                                crrl_metrics["crrl/missing_baseline_count"] = float(missing_baseline_count)
                                raw_advantages = r - p_hats

                            raw_advantages_float = raw_advantages.to(torch.float32)
                            raw_advantages_var = float(raw_advantages_float.var(unbiased=False).detach().cpu().item())
                            p_hats_float = p_hats.to(torch.float32)
                            group_filter_p_hat_sum += float(p_hats_float.sum().detach().cpu().item())
                            group_filter_p_hat_count += int(p_hats_float.numel())
                            batch.batch["crrl_p_hats"] = p_hats_float.to(raw_advantages.device)
                            crrl_metrics["crrl/adv_before_norm"] = raw_advantages.mean().detach().item()
                            crrl_metrics["crrl/raw_advantages/var"] = raw_advantages_var
                            # Group filtering should use raw (pre-normalized) sequence advantages.
                            raw_advantages_for_filter = raw_advantages

                            response_mask = compute_response_mask(batch)
                            advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
                            quantiles = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=advantages.device)
                            q_vals = torch.quantile(advantages, quantiles)
                            crrl_metrics["crrl/adv_after_norm/p10"] = q_vals[0].item()
                            crrl_metrics["crrl/adv_after_norm/p30"] = q_vals[1].item()
                            crrl_metrics["crrl/adv_after_norm/p50"] = q_vals[2].item()
                            crrl_metrics["crrl/adv_after_norm/p70"] = q_vals[3].item()
                            crrl_metrics["crrl/adv_after_norm/p90"] = q_vals[4].item()
                            advantages = advantages.unsqueeze(-1) * response_mask
                            batch.batch["advantages"] = advantages
                            batch.batch["returns"] = advantages

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        if "advantages" not in batch.batch:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    final_prompt_hidden_rows = None
                    final_response_hidden_rows = None
                    final_response_feature_rows = None
                    final_targets = None

                    if crrl_group_filter_enabled:
                        kept_traj_indices, seen_prompts_in_pool, kept_prompts_in_pool = (
                            self._get_nonzero_advantage_indices(
                                batch, zero_eps=advantage_zero_eps, raw_advantages=raw_advantages_for_filter
                            )
                        )
                        zero_prompts_in_pool = seen_prompts_in_pool - kept_prompts_in_pool
                        seen_prompt_count += seen_prompts_in_pool
                        zero_adv_prompt_count += zero_prompts_in_pool

                        if kept_traj_indices:
                            filtered_batch = batch[kept_traj_indices]

                            if estimator_enabled:
                                if estimator_training_rows is None:
                                    raise RuntimeError(
                                        "Estimator is enabled but estimator_training_rows is missing before filtering."
                                    )
                                filtered_prompt_hidden_rows = [
                                    estimator_training_rows["prompt_hidden_rows"][idx] for idx in kept_traj_indices
                                ]
                                filtered_response_hidden_rows = [
                                    estimator_training_rows["response_hidden_rows"][idx] for idx in kept_traj_indices
                                ]
                                filtered_response_feature_rows = [
                                    estimator_training_rows["response_feature_rows"][idx] for idx in kept_traj_indices
                                ]
                                filtered_targets = [
                                    estimator_training_rows["targets"][idx] for idx in kept_traj_indices
                                ]

                            if final_accumulation is not None:
                                existing_final_uids = set(final_accumulation.non_tensor_batch["uid"])
                                dedup_indices = [
                                    i for i, uid in enumerate(filtered_batch.non_tensor_batch["uid"])
                                    if uid not in existing_final_uids
                                ]
                                if dedup_indices:
                                    filtered_batch = filtered_batch[dedup_indices]
                                    if estimator_enabled:
                                        filtered_prompt_hidden_rows = [
                                            filtered_prompt_hidden_rows[i] for i in dedup_indices
                                        ]
                                        filtered_response_hidden_rows = [
                                            filtered_response_hidden_rows[i] for i in dedup_indices
                                        ]
                                        filtered_response_feature_rows = [
                                            filtered_response_feature_rows[i] for i in dedup_indices
                                        ]
                                        filtered_targets = [filtered_targets[i] for i in dedup_indices]
                                else:
                                    filtered_batch = None

                            if filtered_batch is not None:
                                if final_accumulation is None:
                                    final_accumulation = filtered_batch
                                else:
                                    filtered_batch.meta_info.pop("global_token_num", None)
                                    final_accumulation.meta_info.pop("global_token_num", None)
                                    final_accumulation = DataProto.concat([final_accumulation, filtered_batch])

                                if estimator_enabled:
                                    if final_accumulation_estimator_rows is None:
                                        final_accumulation_estimator_rows = {
                                            "prompt_hidden_rows": filtered_prompt_hidden_rows,
                                            "response_hidden_rows": filtered_response_hidden_rows,
                                            "response_feature_rows": filtered_response_feature_rows,
                                            "targets": filtered_targets,
                                        }
                                    else:
                                        final_accumulation_estimator_rows["prompt_hidden_rows"].extend(
                                            filtered_prompt_hidden_rows
                                        )
                                        final_accumulation_estimator_rows["response_hidden_rows"].extend(
                                            filtered_response_hidden_rows
                                        )
                                        final_accumulation_estimator_rows["response_feature_rows"].extend(
                                            filtered_response_feature_rows
                                        )
                                        final_accumulation_estimator_rows["targets"].extend(filtered_targets)

                        num_final_prompts = (
                            len(set(final_accumulation.non_tensor_batch["uid"]))
                            if final_accumulation is not None
                            else 0
                        )

                        alpha = zero_adv_prompt_count / seen_prompt_count if seen_prompt_count > 0 else 0.0
                        crrl_metrics["crrl/group_filter/alpha"] = float(alpha)
                        crrl_metrics["crrl/group_filter/default_br_size"] = float(default_br_size)
                        crrl_metrics["crrl/group_filter/seen_prompts_total"] = float(seen_prompt_count)
                        crrl_metrics["crrl/group_filter/zero_adv_prompts_total"] = float(zero_adv_prompt_count)
                        crrl_metrics["crrl/group_filter/kept_prompts_total"] = float(
                            seen_prompt_count - zero_adv_prompt_count
                        )
                        crrl_metrics["crrl/group_filter/rounds_total"] = float(group_filter_round_count)
                        crrl_metrics["crrl/group_filter/generated_prompts_total"] = float(
                            group_filter_generated_prompts_total
                        )
                        crrl_metrics["crrl/group_filter/generated_prompts_round_current"] = float(
                            current_round_prompt_count
                        )
                        for round_idx, prompt_count in enumerate(group_filter_round_generated_prompt_counts, start=1):
                            crrl_metrics[f"crrl/group_filter/round_{round_idx}_generated_prompts"] = float(
                                prompt_count
                            )

                        if num_final_prompts < target_prompt_batch_size:
                            batch_delta = target_prompt_batch_size - num_final_prompts
                            estimated_br = int((adaptive_beta * batch_delta) / (1.0 - alpha + 1e-6))
                            next_target = min(default_br_size, estimated_br)
                            next_target = max(1, next_target)
                            target_accumulation_size = next_target
                            print(
                                "[CRRL][GroupFilter] "
                                f"effective={num_final_prompts}/{target_prompt_batch_size}, "
                                f"need={batch_delta}, alpha={alpha:.4f}, next_br={next_target}"
                            )
                            with marked_timer("stop_profile", timing_raw):
                                next_step_profile = (
                                    self.global_steps + 1 in self.config.global_profiler.steps
                                    if self.config.global_profiler.steps is not None
                                    else False
                                )
                                self._stop_profiling(
                                    curr_step_profile and not next_step_profile
                                    if self.config.global_profiler.profile_continuous_steps
                                    else curr_step_profile
                                )
                                prev_step_profile = curr_step_profile
                                curr_step_profile = next_step_profile
                            continue

                        final_sample_indices = self._sample_full_prompt_indices(
                            final_accumulation,
                            target_prompt_count=target_prompt_batch_size,
                            rollout_repeat=rollout_repeat,
                        )
                        batch = final_accumulation[final_sample_indices]
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        if estimator_enabled:
                            if final_accumulation_estimator_rows is None:
                                raise RuntimeError(
                                    "Estimator is enabled but no filtered estimator rows were accumulated "
                                    "for final batch."
                                )

                            final_prompt_hidden_rows = [
                                final_accumulation_estimator_rows["prompt_hidden_rows"][idx]
                                for idx in final_sample_indices
                            ]
                            final_response_hidden_rows = [
                                final_accumulation_estimator_rows["response_hidden_rows"][idx]
                                for idx in final_sample_indices
                            ]
                            final_response_feature_rows = [
                                final_accumulation_estimator_rows["response_feature_rows"][idx]
                                for idx in final_sample_indices
                            ]
                            final_targets = [
                                final_accumulation_estimator_rows["targets"][idx] for idx in final_sample_indices
                            ]
                    else:
                        prompt_rows_in_batch = len(batch.batch) // rollout_repeat
                        if self.config.trainer.crrl.enable:
                            crrl_metrics["crrl/group_filter/alpha"] = 0.0
                            crrl_metrics["crrl/group_filter/default_br_size"] = float(default_br_size)
                            crrl_metrics["crrl/group_filter/seen_prompts_total"] = float(prompt_rows_in_batch)
                            crrl_metrics["crrl/group_filter/zero_adv_prompts_total"] = 0.0
                            crrl_metrics["crrl/group_filter/kept_prompts_total"] = float(prompt_rows_in_batch)
                            crrl_metrics["crrl/group_filter/rounds_total"] = 1.0
                            crrl_metrics["crrl/group_filter/generated_prompts_total"] = float(prompt_rows_in_batch)
                            crrl_metrics["crrl/group_filter/generated_prompts_round_current"] = float(
                                prompt_rows_in_batch
                            )
                        if estimator_enabled:
                            if estimator_training_rows is None:
                                raise RuntimeError(
                                    "Estimator is enabled but estimator_training_rows is missing for final batch."
                                )
                            final_prompt_hidden_rows = list(estimator_training_rows["prompt_hidden_rows"])
                            final_response_hidden_rows = list(estimator_training_rows["response_hidden_rows"])
                            final_response_feature_rows = list(estimator_training_rows["response_feature_rows"])
                            final_targets = list(estimator_training_rows["targets"])

                    if self.config.trainer.crrl.enable:
                        final_prompt_rows = len(batch.batch) // rollout_repeat
                        crrl_metrics["crrl/group_filter/final_sampled_prompts"] = float(final_prompt_rows)
                        crrl_metrics["crrl/group_filter/final_sampled_trajectories"] = float(len(batch.batch))
                        if group_filter_reward_count > 0:
                            crrl_metrics["crrl/reward"] = float(group_filter_reward_sum / group_filter_reward_count)
                        else:
                            crrl_metrics["crrl/reward"] = 0.0
                        final_batch_rewards = batch.batch["token_level_scores"].sum(dim=-1).to(torch.float32)
                        crrl_metrics["crrl/reward_final_batch"] = float(
                            final_batch_rewards.mean().detach().cpu().item()
                        )
                        if group_filter_p_hat_count > 0:
                            crrl_metrics["crrl/p_hats"] = float(group_filter_p_hat_sum / group_filter_p_hat_count)
                        else:
                            crrl_metrics["crrl/p_hats"] = 0.0
                        final_batch_p_hats = batch.batch["crrl_p_hats"].to(torch.float32)
                        crrl_metrics["crrl/p_hats_final_batch"] = float(
                            final_batch_p_hats.mean().detach().cpu().item()
                        )

                    if estimator_enabled:
                        final_prompt_hidden_rows = estimator_fit_accumulation_rows["prompt_hidden_rows"]
                        final_response_hidden_rows = estimator_fit_accumulation_rows["response_hidden_rows"]
                        final_response_feature_rows = estimator_fit_accumulation_rows["response_feature_rows"]
                        final_targets = estimator_fit_accumulation_rows["targets"]
                        final_row_count = len(final_targets)
                        if not (
                            final_row_count
                            == len(final_prompt_hidden_rows)
                            == len(final_response_hidden_rows)
                            == len(final_response_feature_rows)
                        ):
                            raise ValueError(
                                "Estimator fit rows are internally inconsistent: "
                                f"targets={final_row_count}, prompt_hidden={len(final_prompt_hidden_rows)}, "
                                f"response_hidden={len(final_response_hidden_rows)}, "
                                f"response_features={len(final_response_feature_rows)}"
                            )
                        if final_row_count == 0:
                            raise RuntimeError("Estimator is enabled but no rollout rows were accumulated for fitting.")

                        crrl_metrics["crrl/adaptive_estimator/fit_rollout_rows_added"] = float(final_row_count)
                        crrl_metrics["crrl/adaptive_estimator/fit_rollout_prompt_groups_added"] = float(
                            final_row_count / max(1, rollout_repeat)
                        )
                        crrl_metrics["crrl/adaptive_estimator/final_train_batch_rows"] = float(len(batch.batch))
                        crrl_metrics["crrl/adaptive_estimator/fit_to_train_batch_row_ratio"] = float(
                            final_row_count / max(1, len(batch.batch))
                        )
                        online_pred_values_np = np.asarray(estimator_online_value_predictions, dtype=np.float32)
                        online_target_values_np = np.asarray(estimator_online_targets, dtype=np.float32)
                        if online_pred_values_np.shape != online_target_values_np.shape:
                            raise ValueError(
                                "Estimator online metric rows are inconsistent: "
                                f"predictions={online_pred_values_np.shape}, targets={online_target_values_np.shape}"
                            )
                        if online_pred_values_np.size != final_row_count:
                            raise ValueError(
                                "Estimator online metric rows do not match fit rows: "
                                f"metrics={online_pred_values_np.size}, fit_rows={final_row_count}"
                            )

                        online_target_errors = online_pred_values_np - online_target_values_np
                        crrl_metrics["crrl/adaptive_estimator/online_target_mae"] = float(
                            np.mean(np.abs(online_target_errors))
                        )
                        crrl_metrics["crrl/adaptive_estimator/online_target_rmse"] = float(
                            np.sqrt(np.mean(np.square(online_target_errors)))
                        )
                        crrl_metrics["crrl/adaptive_estimator/online_target_bias"] = float(
                            np.mean(online_target_errors)
                        )
                        pred_std = float(online_pred_values_np.std())
                        target_std = float(online_target_values_np.std())
                        if online_pred_values_np.size > 1 and pred_std > 1e-8 and target_std > 1e-8:
                            online_target_pearson = float(
                                np.corrcoef(online_pred_values_np, online_target_values_np)[0, 1]
                            )
                        else:
                            online_target_pearson = 0.0
                        if not np.isfinite(online_target_pearson):
                            online_target_pearson = 0.0
                        crrl_metrics["crrl/adaptive_estimator/online_target_pearson"] = online_target_pearson
                        crrl_metrics["crrl/adaptive_estimator/fit_target_mean"] = float(
                            online_target_values_np.mean()
                        )

                        online_rewards_np = np.asarray(estimator_online_rewards, dtype=np.float32)
                        online_raw_advantages_np = np.asarray(estimator_online_raw_advantages, dtype=np.float32)
                        if (
                            online_rewards_np.size != final_row_count
                            or online_raw_advantages_np.size != final_row_count
                        ):
                            raise ValueError(
                                "Estimator online reward/advantage rows do not match fit rows: "
                                f"rewards={online_rewards_np.size}, advantages={online_raw_advantages_np.size}, "
                                f"fit_rows={final_row_count}"
                            )
                        reward_var = float(online_rewards_np.var())
                        adv_var = float(online_raw_advantages_np.var())
                        crrl_metrics["crrl/adaptive_estimator/adv_var_reduction_ratio"] = (
                            adv_var / reward_var if reward_var > 1e-12 else 0.0
                        )
                        crrl_metrics["crrl/adaptive_estimator/pairwise_sign_acc"] = (
                            float(np.mean(estimator_online_pairwise_sign_matches))
                            if estimator_online_pairwise_sign_matches
                            else 0.0
                        )
                        crrl_metrics["crrl/adaptive_estimator/pred_clip_frac_min"] = float(
                            estimator_online_clip_min_count / final_row_count
                        )
                        crrl_metrics["crrl/adaptive_estimator/pred_clip_frac_max"] = float(
                            estimator_online_clip_max_count / final_row_count
                        )
                        crrl_metrics["crrl/adaptive_estimator/pred_clip_min_count"] = float(
                            estimator_online_clip_min_count
                        )
                        crrl_metrics["crrl/adaptive_estimator/pred_clip_max_count"] = float(
                            estimator_online_clip_max_count
                        )
                        crrl_metrics["crrl/adaptive_estimator/pred_clip_min_reward_match_count"] = float(
                            estimator_online_clip_min_reward_match_count
                        )
                        crrl_metrics["crrl/adaptive_estimator/pred_clip_max_reward_match_count"] = float(
                            estimator_online_clip_max_reward_match_count
                        )
                        if estimator_online_clip_min_count > 0:
                            crrl_metrics["crrl/adaptive_estimator/pred_clip_min_reward_match_frac"] = float(
                                estimator_online_clip_min_reward_match_count / estimator_online_clip_min_count
                            )
                        if estimator_online_clip_max_count > 0:
                            crrl_metrics["crrl/adaptive_estimator/pred_clip_max_reward_match_frac"] = float(
                                estimator_online_clip_max_reward_match_count / estimator_online_clip_max_count
                            )

                        estimator_train_prompt_hidden_rows.extend(final_prompt_hidden_rows)
                        estimator_train_response_hidden_rows.extend(final_response_hidden_rows)
                        estimator_train_response_feature_rows.extend(final_response_feature_rows)
                        estimator_train_targets.extend(float(v) for v in final_targets)
                        estimator_train_step_row_counts.append(final_row_count)
                        estimator_retrain_steps += 1
                        fifo_trimmed_steps = 0
                        fifo_trimmed_rows = 0
                        while len(estimator_train_step_row_counts) > estimator_buffer_max_steps:
                            rows_to_trim = estimator_train_step_row_counts.pop(0)
                            fifo_trimmed_steps += 1
                            fifo_trimmed_rows += rows_to_trim
                            estimator_train_prompt_hidden_rows = estimator_train_prompt_hidden_rows[rows_to_trim:]
                            estimator_train_response_hidden_rows = estimator_train_response_hidden_rows[rows_to_trim:]
                            estimator_train_response_feature_rows = estimator_train_response_feature_rows[rows_to_trim:]
                            estimator_train_targets = estimator_train_targets[rows_to_trim:]
                        crrl_metrics["crrl/adaptive_estimator/buffer_rows"] = float(len(estimator_train_targets))
                        crrl_metrics["crrl/adaptive_estimator/buffer_steps"] = float(
                            len(estimator_train_step_row_counts)
                        )
                        crrl_metrics["crrl/adaptive_estimator/fifo_max_steps"] = float(estimator_buffer_max_steps)
                        crrl_metrics["crrl/adaptive_estimator/fifo_trimmed_steps"] = float(fifo_trimmed_steps)
                        crrl_metrics["crrl/adaptive_estimator/fifo_trimmed_rows"] = float(fifo_trimmed_rows)
                        crrl_metrics["crrl/adaptive_estimator/warmup_steps"] = float(estimator_warmup_steps)
                        crrl_metrics["crrl/adaptive_estimator/warmup_remaining_steps"] = float(
                            max(0, estimator_warmup_steps - estimator_retrain_steps)
                        )

                        if estimator_retrain_steps >= estimator_warmup_steps:
                            if estimator_fit_single_fn is None:
                                raise RuntimeError("Adaptive estimator retrain functions are not initialized.")
                            if estimator_fit_config is None or estimator_feature_builder_config is None:
                                raise RuntimeError("Adaptive estimator retrain config is not initialized.")
                            if not estimator_train_targets:
                                raise RuntimeError(
                                    "Adaptive estimator retrain interval reached but no buffered rows exist."
                                )

                            train_target_values = np.asarray(estimator_train_targets, dtype=np.float32)
                            retrain_index = estimator_retrain_count + 1

                            bundle = estimator_fit_single_fn(
                                prompt_hidden_rows=estimator_train_prompt_hidden_rows,
                                response_hidden_rows=estimator_train_response_hidden_rows,
                                response_feature_rows=estimator_train_response_feature_rows,
                                targets=estimator_train_targets,
                                feature_builder_config=estimator_feature_builder_config,
                                fit_config=estimator_fit_config,
                            )
                            estimator = self._build_estimator_from_bundle(bundle)
                            estimator_current_bundle = bundle
                            estimator_current_model_path = None

                            train_predictions = np.asarray(
                                [
                                    estimator.predict_value(
                                        prompt_hidden=prompt_hidden_row,
                                        response_hidden=response_hidden_row,
                                        response_features=response_feature_row,
                                    )
                                    for prompt_hidden_row, response_hidden_row, response_feature_row in zip(
                                        estimator_train_prompt_hidden_rows,
                                        estimator_train_response_hidden_rows,
                                        estimator_train_response_feature_rows,
                                        strict=True,
                                    )
                                ],
                                dtype=np.float32,
                            )
                            train_errors = train_predictions - train_target_values
                            crrl_metrics["crrl/adaptive_estimator/train_mae"] = float(np.mean(np.abs(train_errors)))
                            crrl_metrics["crrl/adaptive_estimator/train_rmse"] = float(
                                np.sqrt(np.mean(np.square(train_errors)))
                            )
                            pred_std = float(train_predictions.std())
                            target_std = float(train_target_values.std())
                            if train_predictions.size > 1 and pred_std > 1e-8 and target_std > 1e-8:
                                pearson = float(np.corrcoef(train_predictions, train_target_values)[0, 1])
                            else:
                                pearson = 0.0
                            if not np.isfinite(pearson):
                                pearson = 0.0
                            crrl_metrics["crrl/adaptive_estimator/pred_target_pearson"] = pearson

                            estimator_retrain_count = retrain_index
                            crrl_metrics["crrl/adaptive_estimator/last_label_mean"] = float(train_target_values.mean())
                            crrl_metrics["crrl/adaptive_estimator/window_rows_used_for_fit"] = float(
                                len(estimator_train_targets)
                            )
                            crrl_metrics["crrl/adaptive_estimator/window_steps_used_for_fit"] = float(
                                len(estimator_train_step_row_counts)
                            )

                    if self.config.trainer.crrl.enable:
                        if prompt_reward_log_dir is None:
                            raise RuntimeError("CRRL prompt reward log directory is not initialized.")
                        with marked_timer("dump_prompt_reward_log", timing_raw, color="green"):
                            self._dump_prompt_reward_log(
                                output_dir=prompt_reward_log_dir,
                                accumulator=prompt_reward_log_accumulator,
                                order=prompt_reward_log_order,
                                group_filter_enabled=crrl_group_filter_enabled,
                                rollout_repeat=rollout_repeat,
                                rounds_total=group_filter_round_count,
                                final_train_batch_rows=len(batch.batch),
                            )
                        crrl_metrics["crrl/prompt_reward_log/prompt_count"] = float(
                            len(prompt_reward_log_order)
                        )
                        crrl_metrics["crrl/prompt_reward_log/trajectory_count"] = float(
                            sum(
                                len(prompt_reward_log_accumulator[uid]["rollout_rewards"])
                                for uid in prompt_reward_log_order
                            )
                        )

                    final_accumulation = None
                    final_accumulation_estimator_rows = None
                    estimator_fit_accumulation_rows = self._new_estimator_row_buffer()
                    estimator_online_value_predictions = []
                    estimator_online_targets = []
                    estimator_online_raw_advantages = []
                    estimator_online_rewards = []
                    estimator_online_pairwise_sign_matches = []
                    estimator_online_clip_min_count = 0
                    estimator_online_clip_max_count = 0
                    estimator_online_clip_min_reward_match_count = 0
                    estimator_online_clip_max_reward_match_count = 0
                    target_accumulation_size = (
                        default_br_size if crrl_group_filter_enabled else target_prompt_batch_size
                    )
                    seen_prompt_count = 0
                    zero_adv_prompt_count = 0
                    group_filter_round_count = 0
                    group_filter_generated_prompts_total = 0
                    group_filter_round_generated_prompt_counts = []
                    group_filter_reward_sum = 0.0
                    group_filter_reward_count = 0
                    group_filter_p_hat_sum = 0.0
                    group_filter_p_hat_count = 0
                    prompt_reward_log_accumulator = {}
                    prompt_reward_log_order = []
                    micro_prompts = batch.non_tensor_batch.get("raw_prompt", None)
                    micro_prompts = [_[0]["content"].strip() for _ in micro_prompts]

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if estimator_enabled:
                        if estimator_update_output_dir is None:
                            raise RuntimeError("Adaptive estimator update output directory is not initialized.")
                        with marked_timer("save_estimator_update", timing_raw, color="green"):
                            self._save_adaptive_estimator_update_snapshot(
                                output_dir=estimator_update_output_dir,
                                estimator_model_path=estimator_current_model_path,
                                estimator_bundle=estimator_current_bundle,
                                retrain_steps=estimator_retrain_steps,
                                retrain_count=estimator_retrain_count,
                                train_targets=estimator_train_targets,
                                train_step_row_counts=estimator_train_step_row_counts,
                            )

                    if self.config.trainer.crrl.enable:
                        if crrl_weighted_sampling:
                            cur_sampled_numbers = []
                            for prompt in micro_prompts:
                                prompt2sampled_number[prompt] += 1
                                cur_sampled_numbers.append(prompt2sampled_number[prompt])

                            cur_sampled_numbers = np.array(cur_sampled_numbers, dtype=np.int32)
                            crrl_metrics["crrl/cur_sampled_number/min"] = cur_sampled_numbers.min()
                            crrl_metrics["crrl/cur_sampled_number/max"] = cur_sampled_numbers.max()
                            crrl_metrics["crrl/cur_sampled_number/mean"] = cur_sampled_numbers.mean()

                        metrics.update(crrl_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        if estimator_enabled:
                            self._set_adaptive_estimator_checkpoint_state(
                                estimator_model_path=estimator_current_model_path,
                                estimator_bundle=estimator_current_bundle,
                                retrain_steps=estimator_retrain_steps,
                                retrain_count=estimator_retrain_count,
                                train_prompt_hidden_rows=estimator_train_prompt_hidden_rows,
                                train_response_hidden_rows=estimator_train_response_hidden_rows,
                                train_response_feature_rows=estimator_train_response_feature_rows,
                                train_targets=estimator_train_targets,
                                train_step_row_counts=estimator_train_step_row_counts,
                            )
                        else:
                            self._clear_adaptive_estimator_checkpoint_state()
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                timing_raw = defaultdict(float)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    logger.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        progress_bar.close()
        logger.close()
