import numpy as np
import os
import sys
import ray
import pandas as pd
import subprocess
import re


@ray.remote(num_cpus=1, num_gpus=8, retry_exceptions=True, memory=1024 * 1024 * 1024 * 64)
def sft_model(config):
    from orchestration.experiment_meta_saver import compute_experiment_hash

    experiment_hash = compute_experiment_hash(config)
    hash_dir = os.path.join("output", experiment_hash)

    train_path = os.path.join(hash_dir, "data", "sft_train.parquet")
    valid_path = os.path.join(hash_dir, "data", "sft.parquet")

    batch_size = config["experiment"]["experiment_params"]["sft_params"]["batch_size"]
    ref_model = config["experiment"]["experiment_params"]["model"]
    lr = config["experiment"]["experiment_params"]["sft_params"]["learning_rate"]
    clip_grad = config["experiment"]["experiment_params"]["sft_params"]["clip_grad"]
    num_epochs = config["experiment"]["experiment_params"]["sft_params"]["num_epochs"]
    weight_decay = config["experiment"]["experiment_params"]["sft_params"]["weight_decay"]
    
    save_path = os.path.join(hash_dir, "sft_model")
    project_name = config["experiment"]["project_name"]
    experiment_name = config["experiment"]["experiment_name"]

    ref_model_size = int(re.search("([0-9]+)B", ref_model).group(1))
    micro_batch_size = max(1, batch_size // 8)
    seq_parallel_size = 2 # needed to enable sequence packing in verl SFT
    if ref_model_size > 14:
        micro_batch_size = 2

    subprocess.run(f"""
    NUM_GPUS_PER_NODE=8
    NUM_NODES=1
    NODE_RANK=0
    MASTER_ADDR=127.0.0.1

    TRAIN_PATH={train_path}
    VALID_PATH={valid_path}
    BATCH_SIZE={batch_size}
    MICRO_BATCH_SIZE={micro_batch_size}
    REF_MODEL={ref_model}
    LR={lr}
    CLIP_GRAD={clip_grad}
    SAVE_PATH={save_path}
    PROJECT_NAME={project_name}
    EXPERIMENT_NAME={experiment_name}
    NUM_EPOCHS={num_epochs}
    SEQ_PARALLEL_SIZE={seq_parallel_size}
    WEIGHT_DECAY={weight_decay}

    ~/sky_workdir/encoding-schemes/sft/run_sft.sh
    """.replace("\n", " "), shell=True, check=True)

    # last step saved to save_path/last