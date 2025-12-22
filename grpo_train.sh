#!/bin/bash
python data_process.py
source ~/.bashrc

ray job submit --address="http://127.0.0.1:8888" \
        --runtime-env=$RUNTIME_SCRIPT_DIR/runtime_env.yaml \
        -- \
        python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$TRAIN_DATA_PATH \
        data.val_files=$VALIDATION_DATA_PATH \
        data.train_batch_size=128 \
        data.max_prompt_length=4096 \
        data.max_response_length=16384 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=/ckpt/checkpoint-sft-7b \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
	actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.n=8 \
	actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        algorithm.kl_ctrl.kl_coef=0.001 \
        custom_reward_function.path=$RUNTIME_SCRIPT_DIR/rewards.py \
        custom_reward_function.name=my_reward_function \
        trainer.critic_warmup=0 \
	trainer.val_before_train=False \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name='' \
        trainer.experiment_name='' \
        trainer.n_gpus_per_node=$HOST_GPU_NUM \
        trainer.nnodes=$HOST_NUM \
        trainer.save_freq=50 \
        trainer.test_freq=50 \
        trainer.default_local_dir=${CKPT_PATH} \
        trainer.total_epochs=4 2>&1 | tee ${current_log_file}

exit_code=$?
exit $exit_code