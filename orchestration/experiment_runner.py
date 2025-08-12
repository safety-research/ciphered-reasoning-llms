import argparse
import yaml
import os
import sys
import ray

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from orchestration.experiment_meta_saver import save_experiment_meta, init_experiment_meta_dict


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


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


def load_experiment_steps(config):
    l_stages = []

    l_stages.append({
        'executor': init_experiment_meta_dict,
        'kwargs': {},
        'name': 'init_experiment_meta_dict'
    })

    for stage in config["stages"]:
        l_stages.append({
            'executor': load_stage_executor(stage),
            'kwargs': stage["executor"]["function_kwargs"],
            'name': stage['name']
        })

    l_stages.append({
        'executor': save_experiment_meta,
        'kwargs': {},
        'name': 'save_experiment_meta'
    })

    return l_stages


def execute_pipeline(config):
    print(f"Executing config \n {config}")

    l_stages = load_experiment_steps(config)

    for stage in l_stages:
        print(f"Running {stage['name']}")
        ray.get(stage['executor'].remote(config, **stage['kwargs']))
        print(f"Done with {stage['name']}")


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
