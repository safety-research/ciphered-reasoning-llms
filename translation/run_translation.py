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

    fn_encoding_scheme = get_encoding_scheme(config["experiment"]["experiment_params"]["encoding_scheme"])

    dataset = get_dataset(config["experiment"]["experiment_params"]["dataset"])

    experiment_hash = compute_experiment_hash(config)
    target_path = os.path.join("output", experiment_hash, "data", "ground_truth_translation.parquet")

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    df = pd.DataFrame({"reference_text": dataset, "translated_text": [fn_encoding_scheme(s) for s in dataset]})
    df.to_parquet(target_path)


@ray.remote(num_cpus=1, num_gpus=4)
def generate_prompted_translation(config):
    from vllm import LLM, SamplingParams

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts import get_translation_prompt
    from utils.vllm import kill_vllm_process

    experiment_hash = compute_experiment_hash(config)

    ground_truth_translation = pd.read_parquet(
        os.path.join("output", experiment_hash, "data", "ground_truth_translation.parquet")
    )

    # Build the prompt
    translation_prompt = get_translation_prompt(config["experiment"]["experiment_params"]["translation_prompt"])

    # TODO(sguo35): few shot exemplars

    n_skipped = 0

    l_inputs = []
    for i, row in ground_truth_translation.iterrows():
        if len(row['reference_text']) > 4000:
            n_skipped += 1
            continue

        assert len(row['translated_text']) < 32000

        l_inputs.append(
            {
                "reference_text": row["reference_text"],
                "gt_translation": row["translated_text"],
                "prompt": [
                    {"role": "system", "content": translation_prompt},
                    {
                        "role": "user",
                        "content": f"Modify the following text according to the provided scheme:\n\n{row['reference_text']}",
                    },
                ],
            }
        )

    print(f"Skipped {n_skipped} rows because they were too long.")

    # Generate the outputs
    llm = LLM(model=config["experiment"]["experiment_params"]["model"], enforce_eager=True, tensor_parallel_size=4, max_num_batched_tokens=65536, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(
        temperature=config["experiment"]["experiment_params"]["sampling_params"]["temperature"],
        max_tokens=512,
        n=config["experiment"]["experiment_params"]["sampling_params"]["n"],
    )

    outputs = llm.chat([r["prompt"] for r in l_inputs], sampling_params=sampling_params, use_tqdm=True)

    l_input_token_lens = [len(o.prompt_token_ids) for o in outputs]
    for i, output in enumerate(outputs):
        l_inputs[i]["model_translations"] = [choice.text for choice in output.outputs]

    # Compute logprobs on GT for perplexity calculations
    logprobs_sampling_params = SamplingParams(
        temperature=config["experiment"]["experiment_params"]["sampling_params"]["temperature"],
        max_tokens=1,
        logprobs=0,
        prompt_logprobs=1,
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
    logprobs = llm.chat(l_logprobs_prompts, sampling_params=logprobs_sampling_params, use_tqdm=True)
    gt_logprobs = [o.prompt_logprobs[l_input_token_lens[i] :] for o in logprobs]
    print(gt_logprobs[0][0])
    gt_logprobs = [[next(iter(l.values())) for l in logprob] for logprob in gt_logprobs]
    gt_logprob_toks = [[l.decoded_token for l in logprob] for logprob in gt_logprobs]
    gt_logprobs = [[l.logprob for l in logprob] for logprob in gt_logprobs]

    for i, gt_logprob in enumerate(gt_logprobs):
        l_inputs[i]["gt_logprobs"] = gt_logprob
        l_inputs[i]["gt_logprob_tokens"] = gt_logprob_toks[i]

    df_output = pd.DataFrame(l_inputs)
    df_output.to_parquet(os.path.join("output", experiment_hash, "data", "prompted_translation.parquet"))

    kill_vllm_process(llm)
