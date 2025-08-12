import numpy as np
import os
import sys
import ray
import pandas as pd


@ray.remote(num_cpus=1)
def generate_ground_truth_translation(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from encoding_schemes import get_encoding_scheme
    from data import get_dataset
    from orchestration.experiment_meta_saver import compute_experiment_hash

    fn_encoding_scheme = get_encoding_scheme(config["experiment"]["encoding_scheme"])

    dataset = get_dataset(config["experiment"]["experiment_params"]["dataset"])

    experiment_hash = compute_experiment_hash(config)
    target_path = os.path.join("output", experiment_hash, "data", "ground_truth_translation.parquet")

    df = pd.DataFrame({"reference_text": dataset, "translated_text": [fn_encoding_scheme(s) for s in dataset]})
    df.to_parquet(target_path)


@ray.remote(num_cpus=1, num_gpus=1)
def generate_prompted_translation(config):
    from vllm import LLM, SamplingParams

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts import get_translation_prompt

    experiment_hash = compute_experiment_hash(config)

    ground_truth_translation = pd.read_parquet(
        os.path.join("output", experiment_hash, "data", "ground_truth_translation.parquet")
    )

    # Build the prompt
    translation_prompt = get_translation_prompt(config["experiment"]["experiment_params"]["translation_prompt"])

    l_inputs = []
    for i, row in ground_truth_translation.iterrows():
        l_inputs.append(
            {
                "reference_text": row["reference_text"],
                "gt_translation": row["translated_text"],
                "prompt": [
                    {"role": "system", "content": translation_prompt},
                    {
                        "role": "user",
                        "content": f"Translate the following text according to the provided scheme:\n\n{row['reference_text']}",
                    },
                ],
            }
        )

    # Generate the outputs
    llm = LLM(model=config["experiment"]["experiment_params"]["model"], enforce_eager=True, disable_log_requests=True)
    sampling_params = SamplingParams(
        temperature=config["experiment"]["experiment_params"]["sampling_params"]["temperature"],
        max_tokens=16384,
        n=config["experiment"]["experiment_params"]["sampling_params"]["n"],
    )

    outputs = llm.generate([r["prompt"] for r in l_inputs], sampling_params=sampling_params)

    l_input_token_lens = [len(o.prompt_token_ids) for o in outputs]
    for i, output in enumerate(outputs):
        l_inputs[i]["model_translations"] = [choice.text for choice in output.choices]

    # Compute logprobs on GT for perplexity calculations
    logprobs_sampling_params = SamplingParams(
        temperature=config["experiment"]["experiment_params"]["sampling_params"]["temperature"],
        max_tokens=1,
        logprobs=0,
        prompt_logprobs=0,
        n=1,
    )
    l_logprobs_prompts = []
    for i, row in enumerate(l_inputs):
        l_logprobs_prompts.append(
            [
                *row["prompt"],
                {
                    "role": "assistant",
                    "content": row["gt_translation"],
                },
            ]
        )
    logprobs = llm.generate(l_logprobs_prompts, sampling_params=logprobs_sampling_params)
    gt_logprobs = [o.prompt_logprobs[l_input_token_lens[i] :] for o in logprobs]

    for i, gt_logprob in enumerate(gt_logprobs):
        l_inputs[i]["gt_logprobs"] = gt_logprob

    df_output = pd.DataFrame(l_inputs)
    df_output.to_parquet(os.path.join("output", experiment_hash, "data", "prompted_translation.parquet"))
