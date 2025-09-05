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
from typing import Union, List, Dict, Optional, Any

import pandas as pd
from openai import OpenAI
from openai.types.fine_tuning import FineTuningJob

from pathlib import Path


@ray.remote(num_cpus=1, num_gpus=8, retry_exceptions=True, memory=1024 * 1024 * 1024 * 512)
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
    warmup_steps_ratio = config["experiment"]["experiment_params"]["sft_params"].get("warmup_steps_ratio", 0.1)
    lr_schedule = config["experiment"]["experiment_params"]["sft_params"].get("lr_schedule", "cosine")
    save_freq = config["experiment"]["experiment_params"]["sft_params"].get("save_freq", -1)

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
    micro_batch_size = min(micro_batch_size, 16)

    seq_parallel_size = 2  # needed to enable sequence packing in verl SFT

    cpu_offload = False

    is_dense_model = re.search("A[0-9]+B", ref_model) is None
    if ref_model_size > 20 and is_dense_model:
        cpu_offload = True

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
    CPU_OFFLOAD={cpu_offload}
    WARMUP_STEPS_RATIO={warmup_steps_ratio}
    LR_SCHEDULE={lr_schedule}
    SAVE_FREQ={save_freq}

    ~/sky_workdir/encoding-schemes/sft/run_sft.sh
    """.replace(
            "\n", " "
        ),
        shell=True,
        check=True,
    )

    # last step saved to save_path/last


def convert_sft_parquet_to_jsonl(parquet_path, output_json_path):
    # 1) Read Parquet and convert to JSONL temp file
    df = pd.read_parquet(parquet_path)

    n_rows_written = 0
    with open(output_json_path, "w", encoding="utf-8") as f:
        for idx, raw in enumerate(df["messages"]):

            json_line = {"messages": list(raw)}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            n_rows_written += 1

    print(f"[prep] Wrote {n_rows_written} training rows to {output_json_path}")


# TODO(sguo35): add validation set support, LR multiplier support
@ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 32)
def openai_sft_model(config, train_parquet_override=None, train_jsonl_override=None, meta_override=None, validation_parquet_template=None,  validation_json_template=None, finetuning_parameters={}, model_override=None):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from sft.sft_runner import convert_sft_parquet_to_jsonl
    from env.openai import set_openai_key

    experiment_name = config["experiment"]["experiment_name"]
    experiment_hash = compute_experiment_hash(config)
    hash_dir = os.path.join("output", experiment_hash)

    model_name = config["experiment"]["experiment_params"]["model"]
    if model_override:
        model_name = model_override

    parquet_path = os.path.join(hash_dir, "data", "sft_train.parquet")
    output_json_path = os.path.join(hash_dir, "data", "sft_train.jsonl")
    model_json_path = os.path.join(hash_dir, "data", "sft_model_meta.json")

    if train_parquet_override:
        parquet_path = train_parquet_override
    if train_jsonl_override:
        output_json_path = train_jsonl_override
    if meta_override:
        model_json_path = meta_override

    poll_interval_seconds = 5

    convert_sft_parquet_to_jsonl(parquet_path, output_json_path)

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

    validation_file_id = None
    if validation_parquet_template:
        validation_parquet_template = validation_parquet_template.replace("__HASH__", experiment_hash)
        validation_json_template = validation_json_template.replace("__HASH__", experiment_hash)

        convert_sft_parquet_to_jsonl(validation_parquet_template, validation_json_template)

        print("[upload] Uploading validation file to OpenAI…")
        try:
            with open(validation_json_template, "rb") as fh:
                uploaded = client.files.create(file=fh, purpose="fine-tune")
            validation_file_id = uploaded.id
        except Exception as e:
            print(e)
            for _ in range(10):
                print("!!!!!!!!!")
            raise e
        print(f"[upload] File uploaded with id: {validation_file_id}")

    # 4) Kick off fine-tuning job
    print(f"[job] Creating fine-tuning job for base model: {model_name}")
    try:
        job: FineTuningJob = client.fine_tuning.jobs.create(
            model=model_name,
            training_file=training_file_id,
            validation_file=validation_file_id,
            hyperparameters={
                "n_epochs": config["experiment"]["experiment_params"].get("sft_params", {}).get("num_epochs", 1),
                "batch_size": config["experiment"]["experiment_params"].get("sft_params", {}).get("batch_size", 64),
                **finetuning_parameters
            },
            seed=42,
            suffix="jeff-encoding-schemes"
        )
    except Exception as e:
        print(e)
        for _ in range(10):
            print("!!!!!!!!!")
        raise e
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


def write_test_sft_data_for_extracting_validation_loss(path):
    l_data = [
        {
            "messages": [
                {
                    "content": "This is a test.",
                    "role": "user"
                },
                {
                    "content": "Hello world!",
                    "role": "assistant"
                }
            ]
        }
        for _ in range(10)
    ]
    pd.DataFrame(l_data).to_parquet(path)


def get_valid_loss_for_openai_job(
    json_path: str | Path,
    *,
    client: Optional[OpenAI] = None,
) -> Optional[float]:
    """
    Read a fine-tuning job_id from a JSON file and return the `full_valid_loss`
    from the checkpoint with the highest step number.

    Args:
        json_path: Path to a JSON file containing at least {"job_id": "..."}.
        api_key: Optional API key to construct a client. If omitted, the client
                 will use the OPENAI_API_KEY environment variable.
        client:   Optionally pass a pre-configured OpenAI client. If provided,
                 `api_key` is ignored.

    Returns:
        The `full_valid_loss` (float) from the checkpoint with the highest step,
        or None if no checkpoints or metric is available.

    Raises:
        FileNotFoundError: If `json_path` does not exist.
        ValueError: If `job_id` is missing or invalid.
        openai.OpenAIError: On API errors.
    """
    from env.openai import set_openai_key

    set_openai_key()

    # Build / reuse client
    client = client or OpenAI()

    # Load job_id
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("JSON must contain a non-empty string field 'job_id'.")

    # List checkpoints (handle pagination)
    checkpoints_iter = client.fine_tuning.jobs.checkpoints.list(
        fine_tuning_job_id=job_id
    )

    best_cp: Optional[dict] = None
    best_step: int = -1

    for cp in checkpoints_iter:
        # Be defensive about field names/types
        step = (
            getattr(cp, "step_number", None)
            or getattr(cp, "step", None)
            or (isinstance(cp, dict) and (cp.get("step_number") or cp.get("step")))
        )
        try:
            step = int(step) if step is not None else -1
        except (TypeError, ValueError):
            step = -1

        if step > best_step and cp.metrics.full_valid_loss is not None:
            best_step = step
            # Normalize to dict for easy access
            best_cp = cp if isinstance(cp, dict) else cp.__dict__

        print(cp)

    if best_cp is None:
        raise Exception(f"Expected best checkpoint to not be None!")

    print(best_cp)
    # Extract metrics.full_valid_loss
    metrics = (
        best_cp.get("metrics")
        if isinstance(best_cp, dict)
        else getattr(best_cp, "metrics", None)
    ) or {}

    print(metrics)
    fvl = metrics.full_valid_loss
    assert fvl is not None
    print(f"job id={job_id} fvl={fvl}")
    for _ in range(10):
        print("!!!!!!")

    return fvl


# TODO(sguo35): implement fireworks ai support
# 1. check if existing fireworks dataset exists
# 2. upsert fireworks dataset

# 3. check if existing model exists
# 4. FT model
# 5. quantize model

# 6. check if deployment exists
# 7. start deployment, (load lora layers??), run prompted
# 8. spin down deployment


@ray.remote(num_cpus=1, memory=16 * 1024 * 1024 * 1024)
def upload_fireworks_dataset(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from sft.sft_runner import convert_sft_parquet_to_jsonl
    from env.fireworks import set_fireworks_api_key

    set_fireworks_api_key()

    experiment_hash = compute_experiment_hash(config)
    hash_dir = os.path.join("output", experiment_hash)

    for suffix in ['', '_train']:
        parquet_path = os.path.join(hash_dir, "data", f"sft{suffix}.parquet")
        output_json_path = os.path.join(hash_dir, "data", f"sft{suffix}.jsonl")

        convert_sft_parquet_to_jsonl(parquet_path, output_json_path)

        dataset_name = f"{experiment_hash}{suffix}_jeff"
        dataset_name = dataset_name.replace("_", "-")

        assert os.system(f"""
        firectl create dataset
        {dataset_name}
        {output_json_path}
        """.replace("\n", "")) == 0


@ray.remote(num_cpus=1, memory=32 * 1024 * 1024 * 1024)
def finetune_model_on_fireworks(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash

    experiment_hash = compute_experiment_hash(config)

    base_model = config["experiment"]["experiment_params"]["model"]

    project_name = config["experiment"]["project_name"]

    train_dataset_name = f"{experiment_hash}_train_jeff"
    train_dataset_name = train_dataset_name.replace("_", "-")

    valid_dataset_name = f"{experiment_hash}_jeff"
    valid_dataset_name = valid_dataset_name.replace("_", "-")

    target_model_name = f"{experiment_hash}-model-jeff"
    job_id = f"{experiment_hash}"

    # TODO(sguo35): check if the model already exists or if we will overwrite if we train with the same model again?

    # TODO(sguo35): patch LR in here
    assert os.system(f"""
    firectl create sftj
    --base-model {base_model}
    --job-id {job_id}
    --dataset {train_dataset_name}
    --output-model {target_model_name}
    --learning-rate 2e-6
    --epochs 1
    --evaluation-dataset {valid_dataset_name}
    --max-context-length 32768
    --lora-rank 32
    --wandb
    --wandb-entity sguo35
    --wandb-api-key {os.environ['WANDB_API_KEY']}
    --wandb-project {project_name}
    """.replace("\n", "")) == 0

    # TODO(sguo35): wait for the model to finish training here

    # TODO(sguo35): write the sft_model_meta.json