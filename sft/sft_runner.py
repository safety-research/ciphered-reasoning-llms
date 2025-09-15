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
from pathlib import Path
import requests
import asyncio

import pandas as pd
import socket
from together import Together

from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util import get_node_ip_address


@ray.remote(num_cpus=2)  # you can also add max_concurrency if desired
class Coordinator:
    def __init__(self):
        self._info = None
        self._event = asyncio.Event()

    async def set(self, addr: str, port: int):
        self._info = (addr, port)
        self._event.set()

    async def get(self, timeout_s: float = 120.0):
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None
        return self._info


def pick_free_port() -> int:
    """Pick an available TCP port on the current node."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@ray.remote(num_cpus=1, num_gpus=8, memory=1024 * 1024 * 1024 * 512)
def sft_model(config, node_rank_override=None, num_nodes_override=None, coord=None, cpu_offload_override=None, micro_batch_size_override=None):
    from orchestration.experiment_meta_saver import compute_experiment_hash
    from sft.sft_runner import pick_free_port

    experiment_hash = compute_experiment_hash(config)
    hash_dir = os.path.join("output", experiment_hash)

    train_path = os.path.join(hash_dir, "data", "sft_train.parquet")
    valid_path = os.path.join(hash_dir, "data", "sft.parquet")

    n_gpus = len(ray.get_gpu_ids())

    node_rank = 0
    num_nodes = 1
    master_addr = "127.0.0.1"
    master_port = pick_free_port()
    
    if coord:
        node_rank = node_rank_override
        num_nodes = num_nodes_override

        if node_rank == 0:
            master_addr = get_node_ip_address()
            master_port = pick_free_port()
            # Publish rendezvous info for the other node
            ray.get(coord.set.remote(master_addr, master_port))
        else:
            result = ray.get(coord.get.remote(timeout_s=300.0))
            if result is None:
                raise RuntimeError("Timed out waiting for master rendezvous info")
            master_addr, master_port = result

        print(
            f"[node_rank={node_rank}] master_addr={master_addr} master_port={master_port} "
            f"num_nodes={num_nodes}"
        )

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

    dp_size = (4 * num_nodes)
    micro_batch_size = max(1, batch_size // dp_size)

    global_max_micro_batch_sz = 16 if num_nodes == 1 else 32
    micro_batch_size = min(micro_batch_size, global_max_micro_batch_sz)
    if micro_batch_size_override is not None:
        micro_batch_size = micro_batch_size_override

    seq_parallel_size = 2  # needed to enable sequence packing in verl SFT

    cpu_offload = False
    if cpu_offload_override is not None:
        cpu_offload = cpu_offload_override

    # is_dense_model = re.search("A[0-9]+B", ref_model) is None
    # if ref_model_size > 20 and is_dense_model:
    #     cpu_offload = True

    while (batch_size // dp_size) % micro_batch_size != 0:
        micro_batch_size -= 1

    subprocess.run(
        f"""
    NCCL_SOCKET_NTHREADS=4
    NCCL_NSOCKS_PERTHREAD=2

    NUM_GPUS_PER_NODE={n_gpus}
    NUM_NODES={num_nodes}
    NODE_RANK={node_rank}
    MASTER_ADDR={master_addr}
    MASTER_PORT={master_port}

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


@ray.remote(num_cpus=1, num_gpus=8, memory=1024 * 1024 * 1024 * 512)
def test_all_reduce_bandwidth(config, node_rank_override=None, num_nodes_override=None, coord=None):
    from sft.sft_runner import pick_free_port

    n_gpus = len(ray.get_gpu_ids())

    node_rank = 0
    num_nodes = 1
    master_addr = "127.0.0.1"
    master_port = pick_free_port()
    
    assert coord is not None
    node_rank = node_rank_override
    num_nodes = num_nodes_override

    if node_rank == 0:
        master_addr = get_node_ip_address()
        master_port = pick_free_port()
        # Publish rendezvous info for the other node
        ray.get(coord.set.remote(master_addr, master_port))
    else:
        result = ray.get(coord.get.remote(timeout_s=300.0))
        if result is None:
            raise RuntimeError("Timed out waiting for master rendezvous info")
        master_addr, master_port = result

    print(
        f"[node_rank={node_rank}] master_addr={master_addr} master_port={master_port} "
        f"num_nodes={num_nodes}"
    )

    subprocess.run(
        f"""
    NUM_GPUS_PER_NODE={n_gpus}
    NUM_NODES={num_nodes}
    NODE_RANK={node_rank}
    MASTER_ADDR={master_addr}
    MASTER_PORT={master_port}
    NCCL_SOCKET_NTHREADS=4
    NCCL_NSOCKS_PERTHREAD=2

    torchrun    --nproc_per_node={n_gpus} \
    --nnodes={num_nodes} \
    --node_rank={node_rank} \
    --master_addr={master_addr} \
    --master_port={master_port} \
    /home/ubuntu/sky_workdir/pytorch-communication-benchmarks/allreduce-loop.py
    """.replace(
            "\n", " "
        ),
        shell=True,
        check=True,
    )


@ray.remote(num_cpus=1)
def multinode_sft_model(config, nnodes = None, detach_pg: bool = False, task_options = {}, do_test_all_reduce_bandwidth=False):
    """
    Submit a single 16-GPU job (2 nodes × 8 GPUs). Safe to call multiple times
    concurrently; each call uses its own PG + coordinator.
    Returns: list of ObjectRefs for the two tasks plus the PG (for optional cleanup).
    """
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from sft.sft_runner import Coordinator, sft_model, test_all_reduce_bandwidth

    assert nnodes is not None

    # Create a per-job placement group with two 8-GPU bundles, hard spread across nodes.
    task_resources = task_options.get('resources', {})
    pg = placement_group(
        bundles=[{"GPU": 8, "CPU": 16, "memory": 512 * 1024 * 1024 * 1024, **task_resources} for _ in range(nnodes)],
        strategy="STRICT_SPREAD",
        lifetime="detached" if detach_pg else None,
    )
    ray.get(pg.ready())

    # Pin the coordinator to bundle 0 (so rank-0 and the coordinator share the same node).
    pg0 = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
        placement_group_capture_child_tasks=False,
    )
    coord = Coordinator.options(scheduling_strategy=pg0).remote()

    fn = test_all_reduce_bandwidth if do_test_all_reduce_bandwidth else sft_model

    l_tasks = [
        fn.options(scheduling_strategy=pg0, **task_options).remote(
            config, node_rank_override=0, num_nodes_override=nnodes, coord=coord
        )
    ]

    for i in range(nnodes - 1):
        l_tasks.append(
            fn.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i + 1,
                    placement_group_capture_child_tasks=False,
                ),
                **task_options
            ).remote(
                config, node_rank_override=i + 1, num_nodes_override=nnodes, coord=coord
            )
        )

    ray.get(l_tasks)


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


def together_retrieve_endpoint_information(endpointId):
    url = f"https://api.together.xyz/v1/endpoints/{endpointId}"

    headers = {"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"}

    response = requests.get(url, headers=headers)

    return response.json()


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


@ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 32)
def together_sft_model(config, train_parquet_override=None, train_jsonl_override=None, meta_override=None, 
                      validation_parquet_template=None, validation_json_template=None, 
                      finetuning_parameters={}, model_override=None):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
    from orchestration.experiment_meta_saver import compute_experiment_hash
    from sft.sft_runner import convert_sft_parquet_to_jsonl, together_retrieve_endpoint_information
    
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
    
    # Initialize Together client
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    client = Together(api_key=api_key)
    
    # # Upload training file
    print("[upload] Uploading training file to Together AI…")
    training_file = client.files.upload(file=output_json_path)
    training_file_id = training_file.id
    print(f"[upload] File uploaded with id: {training_file_id}")
    
    validation_file_id = None
    if validation_parquet_template:
        validation_parquet_template = validation_parquet_template.replace("__HASH__", experiment_hash)
        validation_json_template = validation_json_template.replace("__HASH__", experiment_hash)
        
        convert_sft_parquet_to_jsonl(validation_parquet_template, validation_json_template)
        
        print("[upload] Uploading validation file to Together AI…")
        try:
            validation_file = client.files.upload(file=validation_json_template)
            validation_file_id = validation_file.id
        except Exception as e:
            print(e)
            for _ in range(10):
                print("!!!!!!!!!")
            raise e
        print(f"[upload] File uploaded with id: {validation_file_id}")
    
    # Prepare hyperparameters
    hyperparameters = {
        "n_epochs": config["experiment"]["experiment_params"].get("sft_params", {}).get("num_epochs", 1),
        "learning_rate": config["experiment"]["experiment_params"].get("sft_params", {}).get("learning_rate", 2e-5),
        **finetuning_parameters,
        **config["experiment"]["experiment_params"].get("sft_params", {})
    }
    
    # Create fine-tuning job
    print(f"[job] Creating fine-tuning job for base model: {model_name}")
    try:
        job_params = {
            "model": model_name,
            "training_file": training_file_id,
            "suffix": f"jeff-encoding-schemes-{experiment_hash[:8]}",
            "wandb_api_key": os.environ.get("WANDB_API_KEY", ""),
            "lora": False,
            **hyperparameters
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
            
        # Add any additional hyperparameters from finetuning_parameters
        for key, value in finetuning_parameters.items():
            job_params[key] = value
        
        job = client.fine_tuning.create(**job_params)
        job_id = job.id
    except Exception as e:
        print(e)
        for _ in range(10):
            print("!!!!!!!!!")
        raise e
    print(f"[job] Job created: {job_id} (status: {job.status if hasattr(job, 'status') else 'submitted'})")
    
    # Poll for status
    last_status = None
    seen_events = set()

    for _ in range(100000):
        try:
            # Retrieve job status
            job = client.fine_tuning.retrieve(id=job_id)
            status = job.status if hasattr(job, 'status') else 'unknown'
            
            if status != last_status:
                print(f"[status] {status}")
                last_status = status
            
            # Check for events/logs if available
            if hasattr(job, 'events') and job.events:
                for event in job.events:
                    event_id = f"{event.created_at}_{event.message}"
                    if event_id not in seen_events:
                        seen_events.add(event_id)
                        ts = event.created_at
                        level = event.level
                        message = event.message
                        print(f"[event {ts}] {level}: {message}")
            
            # Check if job is complete
            if status in ("succeeded", "completed", "failed", "cancelled"):
                break
                
            time.sleep(poll_interval_seconds)
        except Exception as e:
            print(f"Error checking job status: {e}")
            time.sleep(poll_interval_seconds)
    
    # Handle final result
    if status not in ("succeeded", "completed"):
        raise RuntimeError(f"Fine-tuning did not succeed: status={status}")

    fine_tuned_model = job.output_name
    
    result = {"fine_tuned_model": fine_tuned_model, "job_id": job_id}
    with open(model_json_path, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, indent=2)
    print(f"[done] Wrote result to {model_json_path}: {result}")


def get_valid_loss_for_together_job(
    json_path: str | Path,
    *,
    client: Optional[Together] = None,
) -> Optional[float]:
    """
    Read a fine-tuning job_id from a JSON file and return the validation loss
    from the Together AI fine-tuning job.
    
    Args:
        json_path: Path to a JSON file containing at least {"job_id": "..."}.
        client: Optionally pass a pre-configured Together client.
    
    Returns:
        The validation loss (float) from the fine-tuning job,
        or None if no validation metrics are available.
    
    Raises:
        FileNotFoundError: If `json_path` does not exist.
        ValueError: If `job_id` is missing or invalid.
    """
    # Build / reuse client
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key and not client:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    client = client or Together(api_key=api_key)
    
    # Load job_id
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)
    
    job_id = payload.job_id
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("JSON must contain a non-empty string field 'job_id'.")
    
    # Retrieve job details
    job = client.fine_tuning.retrieve(id=job_id)
    
    # Extract validation loss if available
    validation_loss = None
    
    # Check various possible locations for validation metrics
    if hasattr(job, 'training_metrics'):
        metrics = job.training_metrics
        if isinstance(metrics, dict):
            validation_loss = metrics.eval_loss or metrics.validation_loss
        elif isinstance(metrics, list) and metrics:
            # Get the last epoch's validation loss
            last_metrics = metrics[-1]
            if isinstance(last_metrics, dict):
                validation_loss = last_metrics.eval_loss or last_metrics.validation_loss
    
    if hasattr(job, 'eval_loss'):
        validation_loss = job.eval_loss
    elif hasattr(job, 'validation_loss'):
        validation_loss = job.validation_loss
    
    if validation_loss is not None:
        print(f"job id={job_id} validation_loss={validation_loss}")
        for _ in range(10):
            print("!!!!!!")
    
    return validation_loss


@ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 16)
def deploy_together_model(config, model_id_override=None, deployment_name_override=None):
    """
    Deploy a fine-tuned model on Together AI with auto-timeout and test it.
    
    Args:
        config: Experiment configuration
        model_id_override: Optional model ID to deploy instead of reading from meta file
        deployment_name_override: Optional deployment name override
    """
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
    from orchestration.experiment_meta_saver import compute_experiment_hash
    from sft.sft_runner import together_retrieve_endpoint_information
    
    experiment_hash = compute_experiment_hash(config)
    hash_dir = os.path.join("output", experiment_hash)
    data_dir = os.path.join(hash_dir, "data")
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Get API key
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    client = Together(api_key=api_key)
    
    # Get model ID from meta file or override
    if model_id_override:
        model_id = model_id_override
    else:
        model_meta_path = os.path.join(data_dir, "sft_model_meta.json")
        if not os.path.exists(model_meta_path):
            raise FileNotFoundError(f"Model meta file not found: {model_meta_path}")
        
        with open(model_meta_path, "r") as f:
            meta = json.load(f)
        model_id = meta.get("fine_tuned_model")
        if not model_id:
            raise ValueError("No fine_tuned_model found in meta file")
    
    print(f"[deploy] Deploying model: {model_id}")
    
    # Create deployment name
    deployment_name = deployment_name_override or f"deploy-{experiment_hash[:8]}-{int(time.time())}"
    
    # Start deployment with 2 min auto timeout on single H100
    try:
        deployment_params = {
            "model": model_id,
            "hardware": "1x_nvidia_h100_80gb_sxm",
            "min_replicas": 1,
            "max_replicas": 1,
            "inactive_timeout": 2,
            "disable_prompt_cache": False,
            "disable_speculative_decoding": True
        }
        
        print(f"[deploy] Creating deployment: {deployment_name}")
        deployment = client.endpoints.create(**deployment_params)
        deployment_id = deployment.id
        deployment_model_path = deployment.name
        
    except Exception as e:
        print(f"[deploy] Error creating deployment: {e}")
        raise e
    
    print(f"[deploy] Deployment created: {deployment_id}")
    
    # Write deployment info to JSON
    deployment_info_path = os.path.join(data_dir, "deployment_info.json")
    deployment_info = {
        "deployment_id": deployment_id,
        "deployment_name": deployment_name,
        "deployment_model_path": deployment_model_path,
        "model_id": model_id,
        "created_at": time.time(),
        "inactive_timeout": 20
    }
    
    with open(deployment_info_path, "w") as f:
        json.dump(deployment_info, f, indent=2)
    print(f"[deploy] Wrote deployment info to {deployment_info_path}")
    
    # Wait for deployment to be ready
    print("[deploy] Waiting for deployment to be ready...")
    max_wait_time = 900
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            # Check deployment status
            deployment = together_retrieve_endpoint_information(deployment_id)
            status = deployment["state"]
            
            if status == 'STARTED':
                print(f"[deploy] Deployment is ready (status: {status})")
                break
            elif status in ["STOPPED", "ERROR", "STOPPING"]:
                raise RuntimeError(f"Deployment failed with status: {status}")
            
            print(f"[deploy] Current status: {status}, waiting...")
            time.sleep(5)
            
        except Exception as e:
            print(f"[deploy] Error checking deployment status: {e}")
            time.sleep(5)
    else:
        raise TimeoutError(f"Deployment did not become ready within {max_wait_time} seconds")
    
    # Test the deployment with "Hello world"
    print("[test] Testing deployment with 'Hello world' prompt...")
    try:
        response = client.chat.completions.create(
            model=deployment_model_path,
            messages=[
                {"role": "user", "content": "Hello world"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        if hasattr(response, 'choices') and response.choices:
            test_response = response.choices[0].message.content
            print(f"[test] Response: {test_response}")
        else:
            print("[test] No response received")
            test_response = None
            
        # Save test result
        test_result = {
            "prompt": "Hello world",
            "response": test_response,
            "timestamp": time.time()
        }
        
        test_result_path = os.path.join(data_dir, "deployment_test_result.json")
        with open(test_result_path, "w") as f:
            json.dump(test_result, f, indent=2)
        print(f"[test] Saved test result to {test_result_path}")
        
    except Exception as e:
        print(f"[test] Error testing deployment: {e}")
        # Continue to shutdown even if test fails


@ray.remote(num_cpus=1, retry_exceptions=True, memory=4 * 1024 * 1024 * 1024)
def shutdown_together_deployment(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
    from orchestration.experiment_meta_saver import compute_experiment_hash
    from sft.sft_runner import together_retrieve_endpoint_information
    
    experiment_hash = compute_experiment_hash(config)
    hash_dir = os.path.join("output", experiment_hash)
    data_dir = os.path.join(hash_dir, "data")

    deployment_info_path = os.path.join(data_dir, "deployment_info.json")
    with open(deployment_info_path, "r") as fp:
        d_deployment_info = json.load(fp)

    deployment_id = d_deployment_info["deployment_id"]

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    client = Together(api_key=api_key)

    # Shutdown deployment
    print("[shutdown] Shutting down deployment...")
    try:
        client.endpoints.delete(deployment_id)
        print(f"[shutdown] Deployment {deployment_id} shutdown initiated")
    except Exception as e:
        raise RuntimeError(f"[shutdown] Error shutting down deployment: {e}")
    
    print("[complete] Deployment lifecycle completed")

