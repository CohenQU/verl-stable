#!/bin/bash

export VLLM_ATTENTION_BACKEND=XFORMERS


clip_ratio_high=0.35
kl_loss_coef=0.001
entropy_coeff=0.001
sft_loss_coeff=0

task=${1}
dataset=${2}
train_batch_size=${3}
n=${4}
ppo_mini_batch_size=${5}
ppo_micro_batch_size=${6}
max_token=${7}


train_file=/project/flame/yuxiaoq/datasets/${dataset}/train.parquet
test_file=/project/flame/yuxiaoq/datasets/deepscalar_RL_test_benchmark/test.parquet
experiment_name=${task}
default_local_dir=/tmp/${task}

project_name=grpo

export WANDB_API_KEY="315e959a539374dbcc7d86cec33ede7187be943f"
export WANDB_ENTITY=yuxiao98

/home/yuxiaoq/workspace/bash/utils/data_sync_in.sh ${task}

mkdir -p /tmp/${task}/models

source /project/flame/yuxiaoq/miniconda3/bin/activate

conda activate verl

/home/yuxiaoq/workspace/bash/utils/data_sync_back.sh ${task} & 

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_file} \
    data.val_files=${test_file} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=/project/flame/yuxiaoq/huggingface/hub/models--d1shs0ap--easy-8k/snapshots/739e47db740132b65e71bf164566679e98193fc1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${ppo_micro_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.only_train_on_positive=False \
    actor_rollout_ref.actor.remove_truncated=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_token} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.sft_loss_coeff=${sft_loss_coeff} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${n} \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=/home/yuxiaoq/workspace/verl-stable/verl/utils/reward_score/curriculum_math/compute_score.py \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_training_steps=300 \
    trainer.default_local_dir=${default_local_dir} \
    trainer.extrapolation_val=False \
    data.max_extrapolation_length=40000 \
    trainer.total_epochs=10

sleep 1200