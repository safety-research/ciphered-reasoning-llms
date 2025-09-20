import argparse
import yaml
import os
import sys
import ray
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from orchestration.experiment_meta_saver import save_experiment_meta, init_experiment_meta_dict, compute_experiment_hash


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    check_experiment_param_failure(config)

    return config


def check_experiment_param_failure(config):
    d_experiment_params = config["experiment"]["experiment_params"]
    if "sft_params" in d_experiment_params and not (d_experiment_params.get("use_sft_model_for_sampling", False) or d_experiment_params.get("use_api_sft_model_for_sampling", False)):
        raise Exception(f"{config}\nConfig is using SFT but does not use SFT model for sampling!")


def load_stage_executor(stage_config):
    import importlib.util
    import sys

    executor_config = stage_config["executor"]
    file_path = executor_config["path"]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = executor_config["function_name"]
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in '{file_path}'.")

    print(f"using function '{function_name}' from '{file_path}'")
    return getattr(module, function_name)


def build_loop_for_stage(stage):
    d_loop_for = stage["loop_for"]

    params = d_loop_for["params"]
    d_template = d_loop_for["template"]

    # assert zip
    param_len = None
    for param in params.values():
        if param_len is None:
            param_len = len(param)
            continue
        
        assert len(param) == param_len

    l_loop_for = []
    for i in range(param_len):
        template = deepcopy(d_template)
        for key in list(template.keys()):
            for param_name, param_values in params.items():
                template[key] = template[key].replace(f"__{param_name}__", str(param_values[i]))

        l_loop_for.append(template)

    stage["loop_for_config"] = stage["loop_for"]
    stage["loop_for"] = l_loop_for


def load_experiment_steps(config):
    l_stages = []

    l_stages.append({"executor": init_experiment_meta_dict, "kwargs": {}, "name": "init_experiment_meta_dict"})

    for stage in config["stages"]:
        if "loop_for" in stage:
            build_loop_for_stage(stage)

        l_stages.append(
            {
                **stage,
                "executor": load_stage_executor(stage),
                "kwargs": stage["executor"]["function_kwargs"],
                "name": stage["name"],
            }
        )

    l_stages.append({"executor": save_experiment_meta, "kwargs": {}, "name": "save_experiment_meta"})

    return l_stages


@ray.remote
def tag_run_for_hash_run():
    return


def execute_pipeline(config):
    l_stages = load_experiment_steps(config)

    experiment_hash = compute_experiment_hash(config)
    checkpoints_dir = os.path.join("output", experiment_hash, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    if config.get("run_for_hash", None) is not None:
        if experiment_hash != config["run_for_hash"]:
            print(f"run_for_hash enabled for {config['run_for_hash']}, skipping {experiment_hash}")
            return
        
        ray.get(tag_run_for_hash_run.remote())

    print(f"Executing config \n {config}")

    for stage in l_stages:
        if stage.get("depends_on", None) is not None:
            prev_stage_dep = stage["depends_on"]

            if not os.path.exists(os.path.join(checkpoints_dir, prev_stage_dep)) and stage.get("skip_stages_without_dep", False):
                print(f"Skipping {stage['name']} because dependency {prev_stage_dep} checkpoint was not found!")
                continue

        print(f"Running {stage['name']}")

        if os.path.exists(os.path.join(checkpoints_dir, stage["name"])):
            if config["experiment"].get("force_overwrite", False) or stage.get("force_overwrite", False):
                print(f"Force overwrite enabled. Overwriting {stage['name']}, hash {experiment_hash}")
            else:
                print(f"Skipping {stage['name']} because there was already a checkpoint, hash {experiment_hash}.")
                continue

        executor = stage["executor"]
        if stage.get("task_options", None) is not None:
            executor = executor.options(**stage["task_options"])
        l_tasks = []
        if stage.get("loop_for", None) is not None:
            l_d_loop_for = stage["loop_for"]

            # partial stage checkpointing not yet supported :(
            for d_loop_for in l_d_loop_for:
                l_tasks.append(executor.remote(config, **stage["kwargs"], **d_loop_for))
        else:
            l_tasks.append(executor.remote(config, **stage["kwargs"]))
        ray.get(l_tasks)
        print(f"Done with {stage['name']}")

        with open(os.path.join(checkpoints_dir, stage["name"]), "w") as fp:
            fp.write("Done.")


# TODO(sguo35): add checkpointing logic so we skip completed steps + overwrite functionality


@ray.remote
def execute_pipeline_remote(config):
    execute_pipeline(config)


if __name__ == "__main__":
    ray.init()

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    config = load_config(args.config)
    execute_pipeline(config)
