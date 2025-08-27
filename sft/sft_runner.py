import numpy as np
import os
import sys
import ray
import pandas as pd
import subprocess
import re
import json
import time
import tempfile
from typing import Union, List, Dict

import pandas as pd
from openai import OpenAI
from openai.types.fine_tuning import FineTuningJob


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
    do_shuffle = config["experiment"]["experiment_params"]["sft_params"].get("shuffle", True)

    save_path = os.path.join(hash_dir, "sft_model")
    project_name = config["experiment"]["project_name"]
    experiment_name = config["experiment"]["experiment_name"]

    dynamic_batch_size_steps = config["experiment"]["experiment_params"]["sft_params"].get("est_num_steps", None)
    if dynamic_batch_size_steps:
        df_train = pd.read_parquet(train_path)
        n_examples = len(df_train)

        batch_size = n_examples / dynamic_batch_size_steps
        batch_size = int(batch_size / 4) * 4

    ref_model_size = int(re.search("([0-9]+)B", ref_model).group(1))
    micro_batch_size = max(1, batch_size // 4)
    micro_batch_size = min(micro_batch_size, 32)

    seq_parallel_size = 2  # needed to enable sequence packing in verl SFT

    is_dense_model = re.search("A[0-9]+B", ref_model) is None
    if ref_model_size > 14 and is_dense_model:
        micro_batch_size = 2

    while (batch_size // 4) % micro_batch_size != 0:
        micro_batch_size -= 1

    subprocess.run(
        f"""
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
    DO_SHUFFLE={do_shuffle}

    ~/sky_workdir/encoding-schemes/sft/run_sft.sh
    """.replace(
            "\n", " "
        ),
        shell=True,
        check=True,
    )

    # last step saved to save_path/last


@ray.remote(num_cpus=1, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def openai_sft_model(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from env.openai import set_openai_key

    experiment_name = config["experiment"]["experiment_name"]
    experiment_hash = compute_experiment_hash(config)
    hash_dir = os.path.join("output", experiment_hash)

    model_name = config["experiment"]["experiment_params"]["model"]
    parquet_path = os.path.join(hash_dir, "data", "sft_train.parquet")
    output_json_path = os.path.join(hash_dir, "data", "sft_train.jsonl")
    model_json_path = os.path.join(hash_dir, "data", "sft_model_meta.json")
    poll_interval_seconds = 5

    # 1) Read Parquet and convert to JSONL temp file
    df = pd.read_parquet(parquet_path)

    n_rows_written = 0
    with open(output_json_path, "w", encoding="utf-8") as f:
        for idx, raw in enumerate(df["messages"]):

            json_line = {"messages": list(raw)}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            n_rows_written += 1

    print(f"[prep] Wrote {n_rows_written} training rows to {output_json_path}")

    set_openai_key()

    # 2) Initialize client
    client = OpenAI()

    # 3) Upload file for fine-tuning
    print("[upload] Uploading training file to OpenAI…")
    # TODO(sguo35)
    with open(output_json_path, "rb") as fh:
        uploaded = client.files.create(file=fh, purpose="fine-tune")
    training_file_id = uploaded.id
    print(f"[upload] File uploaded with id: {training_file_id}")

    # 4) Kick off fine-tuning job
    print(f"[job] Creating fine-tuning job for base model: {model_name}")
    job: FineTuningJob = client.fine_tuning.jobs.create(
        model=model_name,
        training_file=training_file_id,
        hyperparameters={
            "n_epochs": 1,
            "batch_size": 64
        },
        seed=42,
        suffix="jeff-encoding-schemes"
    )
    print(f"[job] Job created: {job.id} (status: {job.status})")

    # 5) Poll for status and stream new events
    seen_event_ids = set()
    last_status = None
    for _ in range(100000):
        try:
            job = client.fine_tuning.jobs.retrieve(job.id)
            status = job.status
            if status != last_status:
                print(f"[status] {status}")
                last_status = status

            # Fetch and print any new events (most recent first in API)
            events = client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job.id, limit=50
            )
            for ev in reversed(events.data):  # print oldest first
                if ev.id in seen_event_ids:
                    continue
                seen_event_ids.add(ev.id)
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev.created_at))
                # ev.data may contain useful details like metrics
                info = ev.message or (ev.data.get("message") if isinstance(ev.data, dict) else None)
                print(f"[event {ts}] {ev.level.upper()}: {info or ev.type}")

            if status in ("succeeded", "failed", "cancelled"):
                break

            time.sleep(poll_interval_seconds)
        except Exception as e:
            print(e)

    # 6) Handle final result and write output JSON
    if job.status != "succeeded":
        raise RuntimeError(f"Fine-tuning did not succeed: status={job.status}")

    if not job.fine_tuned_model:
        # This should be set on success; guard just in case.
        raise RuntimeError("Job succeeded but no fine_tuned_model was returned.")

    result = {"fine_tuned_model": job.fine_tuned_model, "job_id": job.id}
    with open(model_json_path, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, indent=2)
    print(f"[done] Wrote result to {model_json_path}: {result}")