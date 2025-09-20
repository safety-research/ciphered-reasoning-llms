set -x

# Warning! router freeze only works on Qwen3!

torchrun    --nproc_per_node=$NUM_GPUS_PER_NODE \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_PATH" \
    data.val_files=$VALID_PATH \
    data.train_batch_size=$BATCH_SIZE \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.max_length=32768 \
    data.truncation=error \
    data.multiturn.enable=True \
    +data.shuffle=$DO_SHUFFLE \
    model.partial_pretrain=$REF_MODEL \
    model.enable_gradient_checkpointing=True \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    model.fsdp_config.cpu_offload=$CPU_OFFLOAD \
    model.fsdp_config.offload_params=$CPU_OFFLOAD \
    +model.freeze_router_layers=True \
    model.strategy=fsdp \
    ulysses_sequence_parallel_size=$SEQ_PARALLEL_SIZE \
    optim.weight_decay=$WEIGHT_DECAY \
    optim.lr=$LR \
    optim.clip_grad=$CLIP_GRAD \
    +optim.use_stochastic_rounding=True \
    optim.lr_scheduler=$LR_SCHEDULE \
    optim.warmup_steps_ratio=$WARMUP_STEPS_RATIO \
    use_remove_padding=True \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.logger=['console','wandb'] \
    trainer.test_freq=50 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.default_hdfs_dir=null $@