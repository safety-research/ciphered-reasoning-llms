set -x

torchrun    --nproc_per_node=$NUM_GPUS_PER_NODE \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=8008 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VALID_PATH \
    data.train_batch_size=$BATCH_SIZE \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.max_length=32768 \
    data.truncation=error \
    data.multiturn.enable=True \
    model.partial_pretrain=$REF_MODEL \
    model.enable_gradient_checkpointing=True \
    ulysses_sequence_parallel_size=2 \
    optim.weight_decay=0.0 \
    optim.lr=$LR \
    optim.clip_grad=$CLIP_GRAD \
    use_remove_padding=True \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@