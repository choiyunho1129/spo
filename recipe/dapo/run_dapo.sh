#!/usr/bin/env bash
set -xeuo pipefail

project_name='ValueEstimator'
exp_name='Qwen3-4B_DAPO_batch_1024_temp_1.0'

export CUDA_VISIBLE_DEVICES=1,2,3,4
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((2048))
max_response_length=$((8192))
filter_overlong_prompts=False

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 2))
train_prompt_mini_bsz=16
n_resp_per_prompt=8

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

# Paths
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B"}
TRAIN_FILE=${TRAIN_FILE:-"${WORKING_DIR}/data/DAPO-Math-17k-Processed_Splits/all.parquet"}
if [ -n "${TRAIN_FILES:-}" ]; then
    train_files="${TRAIN_FILES}"
else
    train_files="['${TRAIN_FILE}']"
fi
AIME_2024_FILE=${AIME_2024_FILE:-"${WORKING_DIR}/data/AIME_2024.parquet"}
AIME_2025_FILE=${AIME_2025_FILE:-"${WORKING_DIR}/data/AIME_2025.parquet"}
val_files="['${AIME_2024_FILE}', '${AIME_2025_FILE}']"
custom_data_and_reward_path=${CUSTOM_DATA_AND_REWARD_PATH:-"${WORKING_DIR}/recipe/spo/spo_retool.py"}

# Algorithm
temperature=1
top_p=1
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

infer_tp=1 # vllm
train_sp=4 # train
offload=True
rollout_agent_workers=${ROLLOUT_AGENT_WORKERS:-4}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-64}

use_dynamic_bsz=True
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 1))
log_prob_max_token_len_per_gpu=$((actor_max_token_len_per_gpu * 4))

python3 -m recipe.dapo.main_dapo \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.prompt_key=source_prompt \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation='error' \
    data.custom_cls.path="${custom_data_and_reward_path}" \
    data.custom_cls.name=CustomRLHFDataset \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=128 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    custom_reward_function.path="${custom_data_and_reward_path}" \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=dapo \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=$train_sp \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=1500 \
    trainer.val_only=False \
    trainer.val_before_train=True
