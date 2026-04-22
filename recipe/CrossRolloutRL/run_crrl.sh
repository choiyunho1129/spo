set -x
export CUDA_VISIBLE_DEVICES=3
export VLLM_USE_V1=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# wandb stability (override if needed)
export WANDB_API_KEY=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
export WANDB_ENTITY=riasok
export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}
export WANDB_CONSOLE=${WANDB_CONSOLE:-off}
export WANDB_DIR=${WANDB_DIR:-"${OUTPUT_DIR}/wandb"}

#base config
OUTPUT_DIR=${OUTPUT_DIR:-"crrl_verl_pr"}
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-"data/DAPO-Math-17k-Processed_Splits"}
EXP_NAME=${EXP_NAME:-"Qwen3-4B_CRRL_batch_1024_B200"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B"}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
N_VAL=${N_VAL:-8}
DEBUG=${DEBUG:-"False"}

#estimator config
CRRL_MODE=${CRRL_MODE:-"adaptive_estimator"} # non_adaptive_estimator | adaptive_estimator
BASELINE_VALUES=${BASELINE_VALUES:-"data/dapo_math_17k_baseline.jsonl"}
ESTIMATOR_MODEL_PATH=${ESTIMATOR_MODEL_PATH:-"recipe/CrossRolloutRL/estimator/artifacts/qwen3_4b_subset0_1_temp1_pairavg_estimator.joblib"}
ESTIMATOR_FEATURE_BUILDER_CONFIG=${ESTIMATOR_FEATURE_BUILDER_CONFIG:-"recipe/CrossRolloutRL/estimator/single_trajectory_estimator_support/default_feature_builder_config.json"}
ESTIMATOR_FIT_CONFIG=${ESTIMATOR_FIT_CONFIG:-"recipe/CrossRolloutRL/estimator/single_trajectory_estimator_support/default_estimator_fit_config.json"}
ESTIMATOR_PAIR_SIZE=${ESTIMATOR_PAIR_SIZE:-2}
ESTIMATOR_RETRAIN_INTERVAL_STEPS=${ESTIMATOR_RETRAIN_INTERVAL_STEPS:-4}
CRRL_MISSING_PROMPT=${CRRL_MISSING_PROMPT:-"default"} # error | default
CRRL_DEFAULT_P_HAT=${CRRL_DEFAULT_P_HAT:-0.5}
CRRL_WEIGHTED_SAMPLING=${CRRL_WEIGHTED_SAMPLING:-"False"}

# dataset 
train_files="['${TRAIN_DATA_DIR}/all.parquet']"
AIME_2024_FILE=${AIME_2024_FILE:-"data/AIME_2024.parquet"}
AIME_2025_FILE=${AIME_2025_FILE:-"data/AIME_2025.parquet"}
val_files="['${AIME_2024_FILE}', '${AIME_2025_FILE}']"

# tool
tool_config_path=recipe/CrossRolloutRL/crrl_tool_config.yaml

# agent loop
agent_loop_config_path=recipe/CrossRolloutRL/config/crrl_agent.yaml
default_agent_loop=crrl_tool_agent

# file location 
project_name='ValueEstimator'
experiment_name=$EXP_NAME
default_local_dir=$OUTPUT_DIR/$experiment_name/checkpoints
validation_data_dir=$OUTPUT_DIR/$experiment_name/validation_data

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=1
max_prompt_length=2048
max_response_length=$RESPONSE_LENGTH
actor_lr=1e-6

data_prompt_key=prompt
filter_overlong_prompts=True
reward_manager=naive

n_resp_per_prompt=2
n_resp_per_prompt_val=$N_VAL
train_batch_size=512
ppo_mini_batch_size=64
val_batch_size=128
gen_batch_size=512 # prompt batch size before rollout repeat
crrl_enable=True

# Canonicalize mode and enforce supported options.
CRRL_MODE="$(echo "$CRRL_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$CRRL_MODE" = "adaptive_estimator" ]; then
    CRRL_WEIGHTED_SAMPLING=False
elif [ "$CRRL_MODE" != "non_adaptive_estimator" ]; then
    echo "Error: CRRL_MODE must be one of 'non_adaptive_estimator' or 'adaptive_estimator'"
    exit 1
fi

# Match DAPO single-turn preprocessing and scoring: use source_prompt, DAPO reward manager,
# and bypass tool-specific agent configuration entirely.
if [ "$max_turns" -eq 1 ]; then
    data_prompt_key=source_prompt
    filter_overlong_prompts=False
    reward_manager=dapo
    tool_config_path=null
    agent_loop_config_path=null
    default_agent_loop=single_turn_agent
fi

# ================= gpu, vllm config =================
infer_tp=1 # vllm
train_sp=1 # train
offload=True
rollout_agent_workers=${ROLLOUT_AGENT_WORKERS:-4}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-64} 
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

#Vamos! 
python3 -m recipe.CrossRolloutRL.crrl_main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.prompt_key=$data_prompt_key \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=$filter_overlong_prompts \
    data.truncation='error' \
    data.custom_cls.path=recipe/CrossRolloutRL/crrl_retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    +data.gen_batch_size=$gen_batch_size \
    custom_reward_function.path=recipe/CrossRolloutRL/crrl_retool.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=$reward_manager \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
    actor_rollout_ref.rollout.agent.num_workers=$rollout_agent_workers \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.agent.default_agent_loop=$default_agent_loop \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$train_sp \
    trainer.val_before_train=True \
    trainer.log_val_generations=5 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$default_local_dir \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=500 \
    trainer.crrl.enable=$crrl_enable \
    trainer.crrl.mode=$CRRL_MODE \
    trainer.crrl.baseline_values=$BASELINE_VALUES \
    trainer.crrl.missing_prompt=$CRRL_MISSING_PROMPT \
    trainer.crrl.default_p_hat=$CRRL_DEFAULT_P_HAT \
    trainer.crrl.weighted_sampling=$CRRL_WEIGHTED_SAMPLING \
    trainer.crrl.estimator.model_path=$ESTIMATOR_MODEL_PATH \
    trainer.crrl.estimator.feature_builder_config_path=$ESTIMATOR_FEATURE_BUILDER_CONFIG \
    trainer.crrl.estimator.fit_config_path=$ESTIMATOR_FIT_CONFIG \
    trainer.crrl.estimator.pair_size=$ESTIMATOR_PAIR_SIZE \
    trainer.crrl.estimator.retrain_interval_steps=$ESTIMATOR_RETRAIN_INTERVAL_STEPS \
    trainer.debug=$DEBUG  \
