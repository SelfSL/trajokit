#!/bin/bash
# 15-step GSPO training run: 8B, single GPU (Pro 6000 = device 2), ~6-8 hr
set -e
export TRAJOKIT_MAX_TURNS=15
export TRAJOKIT_MAX_CONTEXT=20480          # 4096 prompt + 16384 response
# no TRAJOKIT_ROLLOUT_CACHE: training must generate fresh on-policy rollouts

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.use_kl_loss=False \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.offload_policy=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  data.train_files=swebench_verl.parquet \
  data.val_files=swebench_verl.parquet \
  data.train_batch_size=8 \
  data.max_prompt_length=4096 \
  data.max_response_length=16384 \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=recipes/verl/agent_loop.yaml \
  trainer.total_training_steps=15 \
  trainer.total_epochs=2 \
  trainer.save_freq=5 \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.logger='["console"]' \
  trainer.project_name=trajokit trainer.experiment_name=gspo-8b-train-15step
