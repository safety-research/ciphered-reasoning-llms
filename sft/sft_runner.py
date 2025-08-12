import numpy as np
import os
import sys
import ray
import pandas as pd
import subprocess


@ray.remote(num_cpus=1, num_gpus=4, retry_exceptions=True)
def sft_model(config):
    train_path = # TODO(sguo35)
    valid_path = # TODO(sguo35)

    batch_size = config["experiment"]["experiment_params"]["batch_size"]
    ref_model = config["experiment"]["experiment_params"]["model"]
    lr = config["experiment"]["experiment_params"]["learning_rate"]
    clip_grad = config["experiment"]["experiment_params"]["clip_grad"]
    num_epochs = config["experiment"]["experiment_params"]["num_epochs"]
    
    save_path = # TODO(sguo35)
    project_name = config["experiment"]["project_name"]
    experiment_name = config["experiment"]["experiment_name"]

    # TODO(sguo35): dynamically determine microbatch size & seq parallel to maximize gpu utilization

    subprocess.run(f"""
    NUM_GPUS_PER_NODE=4
    NUM_NODES=1
    NODE_RANK=0
    MASTER_ADDR=127.0.0.1

    TRAIN_PATH={train_path}
    VALID_PATH={valid_path}
    BATCH_SIZE={batch_size}
    MICRO_BATCH_SIZE=1
    REF_MODEL={ref_model}
    LR={lr}
    CLIP_GRAD={clip_grad}
    SAVE_PATH={save_path}
    PROJECT_NAME={project_name}
    EXPERIMENT_NAME={experiment_name}
    NUM_EPOCHS={num_epochs}

    /root/sky_workdir/encoding-schemes/sft/run_sft.sh
    """.replace("\n", " "), shell=True, check=True)

    # last step saved to save_path/last