set -x

export HOME=/home/ubuntu/
# letter to word with dot
# export REF_MODEL=/ext_data/output/c8b98769b65b3721d8fdfe31e96705e29ba8f171/sft_model/last
# identity
# export REF_MODEL=/ext_data/output/ca5c6bf2cb2aae790e5a85fa4674c1e43fcf0e62/sft_model/last
# base64
# export REF_MODEL=/ext_data/output/64309d45fa40be3218975fe871612bea3edad5a0/sft_model/last
# dot between chars
export REF_MODEL=/ext_data/output/5c7a6b2fed62fb54da9a31430cb62af24b8ceb08/sft_model/last

# export REF_MODEL=Qwen/Qwen2.5-3B-Instruct
export PROJECT_NAME=encoding-schemes-rl
# export EXPERIMENT_NAME=lettertowordwithdot-7b-math
# export EXPERIMENT_NAME=identity-7b-math
# export EXPERIMENT_NAME=base64-7b-math
export EXPERIMENT_NAME=dotbetweenchars-7b-math

export VLLM_USE_V1=1

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/dot_between_chars_data/train.parquet \
    data.val_files=$HOME/dot_between_chars_data/val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=$REF_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.5 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$HOME/sky_workdir/encoding-schemes/rl/reward_functions/math_with_adherence.py \
    custom_reward_function.name=math_adherent_and_correct_reward_dot_between_chars \
    reward_model.reward_manager=batch \
    reward_model.launch_reward_fn_async=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.log_val_generations=25 \
    trainer.total_epochs=15 $@