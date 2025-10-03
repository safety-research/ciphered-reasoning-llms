# TODO(sguo35): implement BLEU and perplexity

import evaluate
import sys
import os
import pandas as pd
import ray


@ray.remote(num_cpus=1)
def evaluate_bleu_score(
    config, translation_path_override=None, output_path_override=None
):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from orchestration.experiment_meta_saver import compute_experiment_hash

    bleu = evaluate.load("sacrebleu")

    experiment_hash = compute_experiment_hash(config)

    translation_path = os.path.join(
        "output", experiment_hash, "data", "prompted_translation.parquet"
    )
    if translation_path_override is not None:
        translation_path = translation_path_override.replace(
            "__HASH__", experiment_hash
        )
    df = pd.read_parquet(translation_path)

    l_bleu_scores = []
    for i, row in df.iterrows():
        l_sample_bleus = []

        for n in range(
            config["experiment"]["experiment_params"]["sampling_params"]["n"]
        ):
            l_sample_bleus.append(
                bleu.compute(
                    predictions=[row["model_translations"][n]],
                    references=[row["gt_translation"]],
                )["score"]
            )

        l_bleu_scores.append(l_sample_bleus)

    df["bleu_scores"] = l_bleu_scores

    output_path = os.path.join("output", experiment_hash, "data", "bleu_scores.parquet")
    if output_path_override is not None:
        output_path = output_path_override.replace("__HASH__", experiment_hash)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
