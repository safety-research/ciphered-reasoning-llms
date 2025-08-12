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


@ray.remote(num_cpus=1)
def generate_fewshot_prompt(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash

    experiment_hash = compute_experiment_hash(config)

    target_path = os.path.join("output", experiment_hash, "data", "ground_truth_translation.parquet")
    df = pd.read_parquet(target_path)

    df['len'] = df['translated_text'].map(len)
    df_sample_group = df.sort_values('len').head(100)
    df = df.drop(columns=['len'])

    n_few_shot_examples = config["experiment"]["experiment_params"].get("n_few_shot_examples", 0)

    l_few_shot_examples = []

    for i, row in df.iterrows():
        df_sample = df_sample_group[df_sample_group['translated_text'] != row['translated_text']]
        df_sample = df_sample.sample(n=n_few_shot_examples, random_state=42)

        s = "\n"
        for j, sample_row in df_sample.iterrows():
            s += f"Example {j + 1}. Input: {sample_row['reference_text']} Output: {sample_row['translated_text']}" + "\n"

        l_few_shot_examples.append(s)

    df['few_shot_examples'] = l_few_shot_examples
    df.to_parquet(target_path)


@ray.remote(num_cpus=1, num_gpus=4, retry_exceptions=True)
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


    n_skipped = 0

    l_inputs = []
    for i, row in ground_truth_translation.iterrows():
        if len(row['reference_text']) > 4000:
            n_skipped += 1
            continue

        assert len(row['translated_text']) < 32000

        row_translation_prompt = translation_prompt
        if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
            row_translation_prompt += "\n" + row['few_shot_examples']

        l_inputs.append(
            {
                "reference_text": row["reference_text"],
                "gt_translation": row["translated_text"],
                "prompt": [
                    {"role": "system", "content": row_translation_prompt},
                    {
                        "role": "user",
                        "content": f"Modify the following text according to the provided scheme:\n\n{row['reference_text']}",
                    },
                ],
            }
        )

    print(f"Skipped {n_skipped} rows because they were too long.")

    # Generate the outputs

    sampling_model = config["experiment"]["experiment_params"]["model"]
    if config["experiment"]["experiment_params"].get("use_sft_model_for_sampling", False):
        sampling_model = f"output/{experiment_hash}/sft_model/last"
        print(f"Using SFT model {sampling_model} for translation instead...")

    llm = LLM(model=sampling_model, enforce_eager=True, tensor_parallel_size=4, gpu_memory_utilization=0.7, rope_scaling={"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}, max_model_len=131072)
    sampling_params = SamplingParams(
        temperature=config["experiment"]["experiment_params"]["sampling_params"]["temperature"],
        max_tokens=12000,
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
