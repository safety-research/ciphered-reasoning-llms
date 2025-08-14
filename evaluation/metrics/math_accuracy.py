import sys
import os
import pandas as pd
import ray
import re

from verl.utils.reward_score.math import compute_score, last_boxed_only_string, remove_boxed


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


@ray.remote(num_cpus=1)
def evaluate_math_accuracy(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from orchestration.experiment_meta_saver import compute_experiment_hash
    from evaluation.metrics.math_accuracy import extract_answer

    experiment_hash = compute_experiment_hash(config)
    df = pd.read_parquet(os.path.join("output", experiment_hash, "data", "prompted_cot.parquet"))

    l_correct = []
    for i, row in df.iterrows():
        l_sample_correct = []

        for n in range(config["experiment"]["experiment_params"]["sampling_params"]["n"]):
            try:
                extracted_model_response = extract_answer(row['model_cot'][n])
            except Exception as e:
                print(e)
                l_sample_correct.append(0.0)
                continue

            if len(extracted_model_response) == 0:
                l_sample_correct.append(0.0)
                continue

            if len(row['answer']) == 0:
                l_sample_correct.append(0.0)
                continue

            try:
                l_sample_correct.append(
                    compute_score(extracted_model_response, row['answer'])
                )
            except Exception as e:
                print(e)
                l_sample_correct.append(0.0)
                continue

        l_correct.append(l_sample_correct)

    df["is_corrects"] = l_correct

    df.to_parquet(os.path.join("output", experiment_hash, "data", "math_scores.parquet"))