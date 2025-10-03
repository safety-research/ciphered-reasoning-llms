import sys
import os
import pandas as pd
import ray
import re

from verl.utils.reward_score.math import (
    compute_score,
    last_boxed_only_string,
    remove_boxed,
)


def extract_answer(model_response: str) -> str:
    if "\\boxed" in model_response:
        response = last_boxed_only_string(model_response)
        if response:
            response = remove_boxed(response)
            return response
        else:
            return model_response
    else:
        return model_response


import signal


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


@ray.remote(num_cpus=1)
def evaluate_math_accuracy(config, input_path_override=None, output_path_override=None):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from orchestration.experiment_meta_saver import compute_experiment_hash
    from evaluation.metrics.math_accuracy import extract_answer, timeout

    experiment_hash = compute_experiment_hash(config)

    input_path = os.path.join("output", experiment_hash, "data", "prompted_cot.parquet")
    if input_path_override is not None:
        input_path = input_path_override.replace("__HASH__", experiment_hash)

    df = pd.read_parquet(input_path)

    l_correct = []
    for i, row in df.iterrows():
        l_sample_correct = []

        for n in range(len(row["model_cot"])):
            try:
                with timeout():
                    extracted_model_response = extract_answer(row["model_cot"][n])
            except Exception as e:
                print(e)
                l_sample_correct.append(0.0)
                continue

            if len(extracted_model_response) == 0:
                l_sample_correct.append(0.0)
                continue

            if len(row["answer"]) == 0:
                l_sample_correct.append(0.0)
                continue

            try:
                with timeout():
                    l_sample_correct.append(
                        compute_score(extracted_model_response, row["answer"])
                    )
            except Exception as e:
                print(e)
                l_sample_correct.append(0.0)
                continue

        l_correct.append(l_sample_correct)

    df["is_corrects"] = l_correct

    output_path = os.path.join("output", experiment_hash, "data", "math_scores.parquet")
    if output_path_override is not None:
        output_path = output_path_override.replace("__HASH__", experiment_hash)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
