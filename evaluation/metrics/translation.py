# TODO(sguo35): implement BLEU and perplexity

import evaluate
import sys
import os
import pandas as pd
import ray


@ray.remote(num_cpus=1)
def evaluate_bleu_score(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from orchestration.experiment_meta_saver import compute_experiment_hash

    bleu = evaluate.load("sacrebleu")

    experiment_hash = compute_experiment_hash(config)
    df = pd.read_parquet(os.path.join("output", experiment_hash, "data", "prompted_translation.parquet"))

    l_bleu_scores = []
    for i, row in df.iterrows():
        l_sample_bleus = []

        for n in range(config["experiment"]["experiment_params"]["sampling_params"]["n"]):
            l_sample_bleus.append(
                bleu.compute(
                    predictions=[row["model_translations"][n]],
                    references=[row["gt_translation"]],
                )['score']
            )

        l_bleu_scores.append(l_sample_bleus)

    df["bleu_scores"] = l_bleu_scores

    df.to_parquet(os.path.join("output", experiment_hash, "data", "bleu_scores.parquet"))