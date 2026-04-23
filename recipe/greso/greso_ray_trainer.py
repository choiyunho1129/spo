# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib  # 추가: 고유 ID 생성을 위해 필요
import os
import random  # 추가: 확률적 스킵을 위해 필요
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


def _get_generated_token_stats(batch: DataProto) -> dict[str, int]:
    """Collect valid token stats from one rollout output batch."""
    attention_mask = batch.batch["attention_mask"]
    response_length = batch.batch["responses"].shape[-1]
    prompt_mask = attention_mask[:, :-response_length]
    response_mask = attention_mask[:, -response_length:]

    prompt_tokens = int(prompt_mask.sum().item())
    response_tokens = int(response_mask.sum().item())
    total_tokens = int(attention_mask.sum().item())
    num_sequences = int(attention_mask.shape[0])

    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "num_sequences": num_sequences,
    }


def _compute_raw_sequence_advantages_for_logging(
    batch: DataProto, adv_estimator: AdvantageEstimator
) -> torch.Tensor | None:
    response_mask = batch.batch.get("response_mask", None)
    token_level_rewards = batch.batch.get("token_level_rewards", None)
    if response_mask is None or token_level_rewards is None:
        return None

    response_mask_float = response_mask.to(torch.float32)
    seq_rewards = (token_level_rewards.to(torch.float32) * response_mask_float).sum(dim=-1)

    if adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.GRPO_VECTORIZED):
        uids = batch.non_tensor_batch.get("uid", None)
        if uids is None:
            return None

        uid2indices = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid2indices[str(uid)].append(idx)

        raw_seq_adv = seq_rewards.clone()
        for indices in uid2indices.values():
            idx_tensor = torch.tensor(indices, device=seq_rewards.device, dtype=torch.long)
            mean_val = torch.mean(seq_rewards[idx_tensor])
            raw_seq_adv[idx_tensor] = seq_rewards[idx_tensor] - mean_val
        return raw_seq_adv

    if adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = batch.batch.get("reward_baselines", None)
        if reward_baselines is None:
            return None
        return seq_rewards - reward_baselines.to(torch.float32).to(seq_rewards.device)

    return None


class RayGRESOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        # =================================================================
        # [GRESO] 1. 상태 저장 변수 초기화 (스크립트 config 연동 버전)
        # =================================================================
        self.z_history = defaultdict(int)
        self.is_hard = defaultdict(lambda: True)
        
        # 하드코딩(0.5) 대신 스크립트의 +data 값을 가져옵니다.
        self.p_easy = self.config.data.get('p_easy', 0.5)
        self.p_hard = self.config.data.get('p_hard', 0.5)
        
        # target_zero_variance=0.25 설정을 이용해 논문의 Alpha 값 계산
        target_zv = self.config.data.get('target_zero_variance', 0.25)
        self.alpha_easy = target_zv / 3      # 약 0.083
        self.alpha_hard = target_zv * 2 / 3  # 약 0.167
        
        self.delta_p = 0.01  # 확률 조정 보폭
        
        # 확률 조정을 위한 통계 카운터
        self.n_easy_zero = 0
        self.n_hard_zero = 0
        self.n_total_seen = 0
        # =================================================================

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

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
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        step_generated_prompt_tokens = 0
        step_generated_response_tokens = 0
        step_generated_total_tokens = 0
        step_generated_num_sequences = 0
        
        # =================================================================
        # [GRESO] 추가: 거대한 Pool(버퍼) 및 초기 목표량 세팅
        # =================================================================
        pre_filter_buffer = None
        default_batch = self.config.data.get('default_br_size', 192)
        target_accumulation_size = default_batch
        # 1. 합격자들을 담을 별도의 '금고' 변수를 하나 만듭니다 (함수 시작할 때 초기화)
        if 'final_accumulation' not in locals(): 
            final_accumulation = None
        # =================================================================

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)

                # =================================================================
                # [GRESO] 2. 고정된 UID 부여 및 확률적 필터링 (논문 Eq 3, 4)
                # =================================================================
                # 1) input_ids를 해싱하여 영구적인 고유 ID 생성
                uids = []
                for input_ids in new_batch.batch['input_ids']:
                    uid_str = hashlib.md5(str(input_ids.tolist()).encode()).hexdigest()
                    uids.append(uid_str)
                new_batch.non_tensor_batch["uid"] = np.array(uids, dtype=object)

                # 2) 확률적 스킵 (p_f = 1 - p_e^z_i)
                kept_indices = []
                for idx, uid in enumerate(uids):
                    z_i = self.z_history[uid]
                    if z_i > 0:
                        p_e = self.p_hard if self.is_hard[uid] else self.p_easy
                        p_f = 1.0 - (p_e ** z_i)
                        
                        if random.random() < p_f: # 스킵 확률에 당첨되면 롤아웃 건너뜀
                            continue 
                            
                    kept_indices.append(idx)

                # 만약 배치가 전부 다 스킵되었다면 GPU 연산 없이 바로 다음 데이터로 넘어감!
                if len(kept_indices) == 0:
                    continue
                
                new_batch = new_batch[kept_indices]
                # =================================================================

                # =================================================================
                # [GRESO 1단계-개편]: rollout 전에 prompt만 먼저 누적
                # =================================================================
                if pre_filter_buffer is None:
                    pre_filter_buffer = new_batch
                else:
                    pre_filter_buffer = DataProto.concat([pre_filter_buffer, new_batch])

                current_prompts = len(set(pre_filter_buffer.non_tensor_batch["uid"]))
                if current_prompts < target_accumulation_size:
                    print(f"{current_prompts:} / {target_accumulation_size:} prompts accumulated. Wait for rollout.")
                    continue

                # carry-over 없이, 목표치에 도달한 prompt 풀 전체에 대해 한 번에 rollout 수행
                new_batch = pre_filter_buffer
                pre_filter_buffer = None

                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        gen_stats = _get_generated_token_stats(gen_batch_output)
                        step_generated_prompt_tokens += gen_stats["prompt_tokens"]
                        step_generated_response_tokens += gen_stats["response_tokens"]
                        step_generated_total_tokens += gen_stats["total_tokens"]
                        step_generated_num_sequences += gen_stats["num_sequences"]

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            baseline_stats = _get_generated_token_stats(gen_baseline_output)
                            step_generated_prompt_tokens += baseline_stats["prompt_tokens"]
                            step_generated_response_tokens += baseline_stats["response_tokens"]
                            step_generated_total_tokens += baseline_stats["total_tokens"]
                            step_generated_num_sequences += baseline_stats["num_sequences"]

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    # 수정2: [매우 중요] 기존 랜덤 UUID 삭제
                    # new_batch.non_tensor_batch["uid"] = np.array(
                    #     [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    # )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # =================================================================
                    # [GRESO 2단계-개편]: target prompt pool rollout 결과를 바로 필터링
                    # =================================================================
                    pre_filter_buffer = new_batch

                    metric_name = self.config.algorithm.filter_groups.metric
                    if metric_name == "seq_final_reward":
                        pre_filter_buffer.non_tensor_batch["seq_final_reward"] = (
                            pre_filter_buffer.batch["token_level_rewards"].sum(dim=-1).numpy()
                        )
                    elif metric_name == "seq_reward":
                        pre_filter_buffer.non_tensor_batch["seq_reward"] = (
                            pre_filter_buffer.batch["token_level_scores"].sum(dim=-1).numpy()
                        )

                    prompt_uid2metric_vals = defaultdict(list)
                    for uid, metric_val in zip(
                        pre_filter_buffer.non_tensor_batch["uid"], pre_filter_buffer.non_tensor_batch[metric_name], strict=True
                    ):
                        prompt_uid2metric_vals[uid].append(metric_val)

                    prompt_uid2metric_std = {}
                    for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                        self.n_total_seen += 1 # 실제로 생성해서 확인한 표본 개수
                        std_val = np.std(metric_vals)
                        mean_val = np.mean(metric_vals)
                        prompt_uid2metric_std[prompt_uid] = std_val
                        
                        # z_i 업데이트 및 Easy/Hard 판별
                        if std_val == 0:
                            self.z_history[prompt_uid] += 1
                            is_hard_prompt = (mean_val <= 0.11) 
                            self.is_hard[prompt_uid] = is_hard_prompt
                            
                            if is_hard_prompt: self.n_hard_zero += 1
                            else: self.n_easy_zero += 1
                        else:
                            self.z_history[prompt_uid] = 0 

                    kept_prompt_uids = [
                        uid for uid, std in prompt_uid2metric_std.items()
                        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                    ]
                    
                    num_kept_prompts = len(kept_prompt_uids)
                    kept_traj_idxs = [
                        idx for idx, traj_uid in enumerate(pre_filter_buffer.non_tensor_batch["uid"])
                        if traj_uid in kept_prompt_uids
                    ]

                    # 필터링이 완료된 알짜배기 데이터
                    filtered_batch = pre_filter_buffer[kept_traj_idxs]
                    prompt_bsz = self.config.data.train_batch_size # 목표치 (예: 128)

                    # =================================================================
                    # [GRESO 3단계]: 합격자를 금고에 넣고, 임시 버퍼는 비우기
                    # =================================================================
                    # 1. 합격자를 final_accumulation(금고)에 추가
                    if final_accumulation is None:
                        final_accumulation = filtered_batch
                    else:
                        final_accumulation = DataProto.concat([final_accumulation, filtered_batch])
                    
                    # 2. 임시 바구니는 임무를 다했으니 무조건 초기화 (메모리 누수 방지)
                    pre_filter_buffer = None

                    # 3. 금고에 모인 고유 프롬프트 개수 확인
                    num_final_prompts = len(set(final_accumulation.non_tensor_batch["uid"]))

                    if num_final_prompts < prompt_bsz:
                        B_delta = prompt_bsz - num_final_prompts
                        
                        total_zero = self.n_easy_zero + self.n_hard_zero
                        alpha = total_zero / self.n_total_seen if self.n_total_seen > 0 else 0.2
                        
                        beta = self.config.data.get('beta', 1.25)
                        estimated_Br = int((beta * B_delta) / (1.0 - alpha + 1e-6))
                        
                        default_br_size = self.config.data.get('default_br_size', 384)
                        Br = min(default_br_size, estimated_Br)
                        Br = max(Br, default_batch/3) # 최소 64개는 뽑아오도록 방어
                    
                        print(f"[GRESO] Status: {num_final_prompts}/{prompt_bsz} valid. Need {B_delta} more. Next Pool Target: +{Br}")
                    
                        # 다음 목표치는 '새로 뽑아올 양(Br)'으로만 설정하고 루프 상단으로 이동
                        target_accumulation_size = Br
                        self.gen_steps += 1
                        continue 
                        
                    else:
                        # =================================================================
                        # [GRESO 4단계]: 금고에 128개 이상 꽉 찼음! PPO 업데이트로 넘겨줌!
                        # =================================================================
                        traj_bsz = prompt_bsz * self.config.actor_rollout_ref.rollout.n
                        batch = final_accumulation[:traj_bsz] # 금고에서 딱 필요한 만큼만 잘라냄

                        # [수정] Config에서 상한/하한값 가져오기
                        min_p = self.config.data.get('min_p', 0.05)
                        max_p = self.config.data.get('max_p', 0.95)
                        actual_easy_ratio = 0.0
                        actual_hard_ratio = 0.0

                        # 탐색 확률 자가 조절
                        if self.n_total_seen > 0:
                            actual_easy_ratio = self.n_easy_zero / self.n_total_seen
                            actual_hard_ratio = self.n_hard_zero / self.n_total_seen
                            
                            # Easy 확률 조절 (하드코딩 제거, min_p/max_p 연동)
                            if actual_easy_ratio >= self.alpha_easy:
                                self.p_easy = max(min_p, self.p_easy - self.delta_p)
                            else:
                                self.p_easy = min(max_p, self.p_easy + self.delta_p)
                                
                            # Hard 확률 조절 (하드코딩 제거, min_p/max_p 연동)
                            if actual_hard_ratio >= self.alpha_hard:
                                self.p_hard = max(min_p, self.p_hard - self.delta_p)
                            else:
                                self.p_hard = min(max_p, self.p_hard + self.delta_p)
                        
                        # [중요] 초기화 전 WandB 로깅용 임시 저장 (차트 0 방지)
                        temp_easy = self.n_easy_zero
                        temp_hard = self.n_hard_zero
                        temp_total = self.n_total_seen
                        
                        self.n_easy_zero = 0
                        self.n_hard_zero = 0
                        self.n_total_seen = 0
                        final_accumulation = None # 업데이트로 넘어가므로 금고도 비워줌
                        target_accumulation_size = default_batch

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        # `rollout_corr_helper` only exists in newer verl versions.
                        # Import lazily so the default path still works on older checkouts
                        # where rollout correction is not configured.
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        raw_adv_estimator = self.config.algorithm.adv_estimator
                        try:
                            raw_adv_estimator = AdvantageEstimator(raw_adv_estimator)
                        except ValueError:
                            raw_adv_estimator = None

                        if raw_adv_estimator is not None:
                            raw_seq_adv = _compute_raw_sequence_advantages_for_logging(
                                batch=batch,
                                adv_estimator=raw_adv_estimator,
                            )
                            if raw_seq_adv is not None:
                                metrics["greso/raw_advantages/var"] = raw_seq_adv.var(unbiased=False).detach().item()

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

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
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
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

                # collect metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics["perf/step_generated_total_num_tokens"] = step_generated_total_tokens
                metrics["perf/step_generated_total_num_prompt_tokens"] = step_generated_prompt_tokens
                metrics["perf/step_generated_total_num_response_tokens"] = step_generated_response_tokens
                metrics["perf/step_generated_total_num_sequences"] = step_generated_num_sequences
                if metrics["perf/total_num_tokens"] > 0:
                    metrics["perf/step_generated_to_trained_token_ratio"] = (
                        step_generated_total_tokens / metrics["perf/total_num_tokens"]
                    )
                # [GRESO] 전용 로깅 지표 추가
                ### 시점: 여기 ###
                metrics.update({
                    "greso/p_easy": self.p_easy,
                    "greso/p_hard": self.p_hard,
                    "greso/n_total_seen": temp_total,
                    "greso/n_easy_zero": temp_easy,
                    "greso/n_hard_zero": temp_hard,
                    "greso/alpha_easy": actual_easy_ratio,
                    "greso/alpha_hard": actual_hard_ratio,
                    "greso/target_alpha_easy": self.alpha_easy,
                    "greso/target_alpha_hard": self.alpha_hard,
                })
                if temp_total > 0:
                    metrics["greso/total_skip_ratio"] = (temp_easy + temp_hard) / temp_total
                
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                step_generated_prompt_tokens = 0
                step_generated_response_tokens = 0
                step_generated_total_tokens = 0
                step_generated_num_sequences = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
    # -------------------------------------------------------------------------
    # [GRESO] Override: Checkpoint Save / Load
    # -------------------------------------------------------------------------
    def _save_checkpoint(self):
        # 1. 부모 클래스(RayPPOTrainer)의 기존 저장 로직 먼저 실행
        super()._save_checkpoint()
        
        # 2. 부모가 폴더를 생성했으므로, 같은 경로에 greso 상태를 추가로 저장
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, 
            f'global_step_{self.global_steps}'
        )
        
        # 3. 저장할 데이터 구성
        greso_states = {
            'z_history': dict(self.z_history),  # defaultdict를 일반 dict로 변환
            'is_hard': dict(self.is_hard),
            'p_easy': self.p_easy,
            'p_hard': self.p_hard,
        }
        
        # 4. 파일로 저장
        greso_path = os.path.join(local_global_step_folder, 'greso_states.pt')
        torch.save(greso_states, greso_path)
        print(f"[GRESO] Custom states saved at {greso_path}")
        
    def _load_checkpoint(self):
        # 1. 부모 클래스의 기존 로드 로직 실행 (성공 시 self.global_steps 업데이트 됨)
        super()._load_checkpoint()
        
        # 2. 처음부터 학습하는 경우 (로드된 게 없으면) 안전하게 스킵
        if self.global_steps == 0:
            return
            
        # 3. 로드할 폴더 찾기 (부모 로직과 완벽히 동일한 규칙 적용)
        if self.config.trainer.resume_mode == 'auto':
            from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        else:
            global_step_folder = self.config.trainer.resume_mode
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        
        # 4. GRESO 상태 파일이 있으면 복구
        if global_step_folder:
            greso_path = os.path.join(global_step_folder, 'greso_states.pt')
            if os.path.exists(greso_path):
                states = torch.load(greso_path)
                
                # 기존 defaultdict에 값 업데이트 (기존 초기화된 객체를 덮어쓰지 않고 update)
                self.z_history.update(states.get('z_history', {}))
                self.is_hard.update(states.get('is_hard', {}))
                
                # 확률값 복구 (저장된 게 없으면 현재 설정값 유지)
                self.p_easy = states.get('p_easy', self.p_easy)
                self.p_hard = states.get('p_hard', self.p_hard)
                
                print(f"[GRESO] Successfully restored states from {greso_path}")
                print(f"[GRESO] Restored Z-history count: {len(self.z_history)}")
            else:
                print(f"[GRESO] Warning: No greso_states.pt found in {global_step_folder}")
