import os
import sys
import json
import itertools
import subprocess
import argparse
import re
import ray
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import duckdb
import pandas as pd
import numpy as np
import asyncio


def load_params(json_path: str) -> Dict[str, List[Any]]:
    """Load parameters from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def save_grid(grid: List[Dict[str, Any]], output_path: str):
    """Save the parameter grid to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"\nSaved parameter grid to {output_path}")


def parse_hash_from_output(output: str) -> Optional[str]:
    """Parse hash from eval_orchestrator.py output.

    Looks for pattern <hash>HASH_HERE</hash> in the output.
    Returns None if no hash is found.
    """
    match = re.search(r"<hash>(.*?)</hash>", output)
    return match.group(1) if match else None


@ray.remote
def run_eval_orchestrator_remote(
    params: Dict[str, Any], eval_script: str = "eval_orchestrator.py", dry_run: bool = False
) -> Tuple[str, Optional[str]]:
    """Remote function to run eval_orchestrator.py with the given parameters.

    Returns:
        Tuple of (command output, hash value if found)
    """
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

    # TODO(sguo35): replace this function
    # TODO(sguo35): read the target yaml and recursively replace any params that are in the json grid
    # TODO(sguo35): connect to central ray cluster
    # TOOO(sguo35): request CPU & GPU & sonnet resources
    from evaluation.orchestration.eval_orchestrator import main as eval_orchestrator_main, get_argparser

    cmd = []
    for key, value in params.items():
        if len(str(value)) == 0:
            continue

        if value == 0:
            continue

        l_args = [f"--{key}"]
        if value != 1:
            l_args.append(str(value))
        cmd.extend(l_args)

    print(f"Running command: {' '.join(cmd)}")
    if dry_run:
        return None, None

    parser = get_argparser()
    args = parser.parse_args(cmd)
    hash_value = asyncio.run(eval_orchestrator_main(args))
    return "", hash_value


def generate_param_combinations(
    params: Dict[str, List[Any]], zip_groups: Optional[List[List[str]]] = None
) -> List[Dict[str, Any]]:
    """Generate parameter combinations, with multiple groups of parameters zipped together.

    Args:
        params: Dictionary of parameter lists
        zip_groups: List of lists, where each inner list contains parameters to zip together.
                   For example: [['a', 'b'], ['c', 'd']] will zip a with b and c with d,
                   then take the cross product of these zipped groups.
    """
    if "sql_filter" in params:
        l_sql_filters = params["sql_filter"]
        del params["sql_filter"]
    else:
        l_sql_filters = []

    if not zip_groups:
        # Take cross product of all parameters
        keys = list(params.keys())
        values = list(params.values())
        combinations = list(itertools.product(*values))

        result = [dict(zip(keys, combo)) for combo in combinations]

        df_results = pd.DataFrame(result)
        if len(l_sql_filters) > 0:
            l_sql_filters = [f"({filter})" for filter in l_sql_filters]
            df_results = duckdb.query(f"SELECT * FROM df_results WHERE {'AND'.join(l_sql_filters)}").to_df()
        df_results.to_csv("df_results.csv", index=False)

        return df_results.to_dict(orient="records")

    # First, verify all parameters in zip groups exist and have same length within each group
    for group in zip_groups:
        group_lengths = [len(params[param]) for param in group]
        if len(set(group_lengths)) != 1:
            raise ValueError(f"All parameters in group {group} must have the same length")

    # Create combinations for each zip group
    zipped_groups = []
    for group in zip_groups:
        zipped_combos = list(zip(*[params[param] for param in group]))
        zipped_groups.append(zipped_combos)

    # Get remaining parameters (not in any zip group)
    zipped_params = set(param for group in zip_groups for param in group)
    remaining_params = {k: v for k, v in params.items() if k not in zipped_params}

    # Generate combinations for remaining parameters
    remaining_combos = []
    if remaining_params:
        remaining_keys = list(remaining_params.keys())
        remaining_values = list(remaining_params.values())
        remaining_combos = list(itertools.product(*remaining_values))

    # Combine all combinations
    result = []
    if remaining_combos:
        # If we have remaining parameters, take cross product with all zipped groups
        for remaining_combo in remaining_combos:
            remaining_dict = dict(zip(remaining_keys, remaining_combo))

            # Take cross product of all zipped groups
            for zipped_combos in itertools.product(*zipped_groups):
                combo_dict = remaining_dict.copy()
                for group, combo in zip(zip_groups, zipped_combos):
                    combo_dict.update(dict(zip(group, combo)))
                result.append(combo_dict)
    else:
        # If no remaining parameters, just take cross product of zipped groups
        for zipped_combos in itertools.product(*zipped_groups):
            combo_dict = {}
            for group, combo in zip(zip_groups, zipped_combos):
                combo_dict.update(dict(zip(group, combo)))
            result.append(combo_dict)

    df_results = pd.DataFrame(result)
    if len(l_sql_filters) > 0:
        l_sql_filters = [f"({filter})" for filter in l_sql_filters]
        df_results = duckdb.query(f"SELECT * FROM df_results WHERE {'AND'.join(l_sql_filters)}").to_df()
    df_results.to_csv("df_results.csv", index=False)

    return df_results.to_dict(orient="records")


def patch_param_vars(params: Dict[str, Any]) -> None:
    if "variables" in params:
        for variable in params["variables"]:
            env_var = variable.replace("$", "")
            env_var = os.environ[env_var]

            for key, combinations in params.items():
                for i, combo in enumerate(combinations):
                    if type(combo) == str and variable in combo:
                        print(f"Patching {key}[{i}] from {combo} to {env_var}")
                        params[key][i] = params[key][i].replace(variable, env_var)

        del params["variables"]


def parse_zip_groups(zip_arg: Optional[str]) -> Optional[List[List[str]]]:
    """Parse the zip argument into groups of parameters."""
    if not zip_arg:
        return None
    return [group.split() for group in zip_arg.split("|")]


def main():
    parser = argparse.ArgumentParser(description="Run grid search with eval_orchestrator.py")
    parser.add_argument("json_path", help="Path to JSON file containing parameter lists")
    parser.add_argument(
        "--zip",
        help="Parameters to zip together, separated by |. Each group is space-separated. "
        "Example: 'a b|c d' will zip a with b and c with d",
    )
    parser.add_argument("--eval-script", default="eval_orchestrator.py", help="Path to eval_orchestrator.py")
    parser.add_argument("--output", help="Path to save the generated parameter grid as JSON")
    parser.add_argument("--num-cpus", type=int, help="Number of CPUs to use for parallel execution")
    parser.add_argument("--dry-run", action="store_true", help="Dry run the script")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["GLOG_logtostderr"] = "1"

    # Initialize Ray
    ray.init(num_cpus=args.num_cpus if args.num_cpus else None)

    try:
        # Load parameters
        params = load_params(args.json_path)

        # Parse zip groups
        zip_groups = parse_zip_groups(args.zip)

        patch_param_vars(params)

        # Generate combinations
        combinations = generate_param_combinations(params, zip_groups)

        np.random.shuffle(combinations)

        # Run eval_orchestrator for each combination in parallel using Ray
        futures = []
        for i, combo in enumerate(combinations, 1):
            print(f"\nScheduling combination {i}/{len(combinations)}")
            print(f"Parameters: {combo}")
            future = run_eval_orchestrator_remote.remote(combo, args.eval_script, args.dry_run)
            futures.append((combo, future))

        # Collect results
        for combo, future in futures:
            output, hash_value = ray.get(future)
            if hash_value:
                combo["hash"] = hash_value
                print(f"Found hash: {hash_value}")
            else:
                print("Warning: No hash found in output")

        # Save grid if output path is specified
        if args.output and not args.dry_run:
            save_grid(combinations, args.output)

    finally:
        # Shutdown Ray
        ray.shutdown()


if __name__ == "__main__":
    main()
