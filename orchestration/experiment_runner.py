import argparse
import yaml
import os
import sys
import ray

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from orchestration.experiment_meta_saver import save_experiment_meta


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
    raw_fn = getattr(module, function_name)

    function_kwargs = dict(executor_config.get("function_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **function_kwargs)

    return wrapped_fn


def load_experiment_steps(config):
    l_stages = []
    l_stage_kwargs = []

    for stage in config["stages"]:
        l_stages.append(load_stage_executor(stage))
        l_stage_kwargs.append(config["executor"]["function_kwargs"])

    l_stages.append(save_experiment_meta)
    l_stage_kwargs.append({})

    return l_stages, l_stage_kwargs


def execute_pipeline(args):
    config = load_config(args.config)
    l_stages, l_stage_kwargs = load_experiment_steps(config)

    for stage, stage_kwargs in zip(l_stages, l_stage_kwargs):
        ray.get(stage.remote(config, **stage_kwargs))


@ray.remote
def execute_pipeline_remote(args):
    execute_pipeline(args)


if __name__ == "__main__":
    ray.init()

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    execute_pipeline(args)
