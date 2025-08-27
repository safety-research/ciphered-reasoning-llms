import numpy as np
import os
import sys
import ray
import pandas as pd
import tiktoken
import json
import re
import asyncio


encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens_from_messages(messages):
    total_tokens = 0
    for msg in messages:
        # Count role and content separately
        total_tokens += len(encoding.encode(msg.get("role", "")))
        total_tokens += len(encoding.encode(msg.get("content", "")))
    return total_tokens


@ray.remote(num_cpus=1)
def generate_ground_truth_translation(config, dataset_override=None):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from encoding_schemes import get_encoding_scheme, is_async_encoding_scheme
    from data import get_dataset
    from orchestration.experiment_meta_saver import compute_experiment_hash

    fn_encoding_scheme = get_encoding_scheme(config["experiment"]["experiment_params"]["encoding_scheme"], config)

    dataset_name = config["experiment"]["experiment_params"]["dataset"]
    if dataset_override:
        dataset_name = dataset_override
    dataset = get_dataset(dataset_name)


    experiment_hash = compute_experiment_hash(config)
    target_path = os.path.join("output", experiment_hash, "data", "ground_truth_translation.parquet")

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    async def gather_all(tasks):
        return await asyncio.gather(*tasks)

    translated_text = [fn_encoding_scheme(s) for s in dataset]
    if is_async_encoding_scheme(config["experiment"]["experiment_params"]["encoding_scheme"]):
        translated_text = asyncio.run(gather_all(translated_text))

    # Note that translated is the reference input and the English is the translated target!
    df = pd.DataFrame({"reference_text": translated_text, "translated_text": dataset})

    if config["experiment"]["experiment_params"].get("validation_set_frac", 0):
        validation_set_frac = config["experiment"]["experiment_params"]["validation_set_frac"]
        train_set_frac = 1.0 - validation_set_frac

        df_train = df.sample(frac=train_set_frac, random_state=42)
        df_valid = df[~df.index.isin(df_train.index)]

        train_path = os.path.join("output", experiment_hash, "data", "ground_truth_translation_train.parquet")
        df_train.to_parquet(train_path)
        df_valid.to_parquet(target_path)
    else:
        df.to_parquet(target_path)


def get_few_shot_examples(df, df_sample_group, config):
    n_few_shot_examples = config["experiment"]["experiment_params"].get("n_few_shot_examples", 0)

    l_few_shot_examples = []

    for i, row in df.iterrows():
        df_sample = df_sample_group[df_sample_group["translated_text"] != row["translated_text"]]
        df_sample = df_sample.sample(n=n_few_shot_examples, random_state=42)

        s = "\n"
        for j, sample_row in df_sample.iterrows():
            s += (
                f"Example {j + 1}. Input: {sample_row['reference_text']} Output: {sample_row['translated_text']}" + "\n"
            )

        l_few_shot_examples.append(s)

    return l_few_shot_examples


@ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 32)
def generate_fewshot_prompt(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from translation.run_translation import get_few_shot_examples

    experiment_hash = compute_experiment_hash(config)

    l_suffixes = [""]
    if config["experiment"]["experiment_params"].get("validation_set_frac", 0):
        l_suffixes.append("_train")

    for suffix in l_suffixes:
        target_path = os.path.join("output", experiment_hash, "data", f"ground_truth_translation{suffix}.parquet")
        df = pd.read_parquet(target_path)

        df["len"] = df["translated_text"].map(len)
        df_sample_group = df.sort_values("len").head(100)
        df = df.drop(columns=["len"])

        df["few_shot_examples"] = get_few_shot_examples(df, df_sample_group, config)
        df.to_parquet(target_path)


@ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 32)
def generate_sft_dataset(config, skip_too_long=True):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts import get_translation_prompt
    from translation.run_translation import count_tokens_from_messages

    experiment_hash = compute_experiment_hash(config)

    for suffix in ["", "_train"]:
        ground_truth_translation = pd.read_parquet(
            os.path.join("output", experiment_hash, "data", f"ground_truth_translation{suffix}.parquet")
        )

        # Build the prompt
        translation_prompt = get_translation_prompt(config["experiment"]["experiment_params"]["translation_prompt"])

        n_skipped = 0

        l_inputs = []
        for i, row in ground_truth_translation.iterrows():
            if len(row["reference_text"]) > 4000 and skip_too_long:
                n_skipped += 1
                continue

            row_translation_prompt = translation_prompt
            if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
                row_translation_prompt += "\n" + row["few_shot_examples"]

            l_inputs.append(
                {
                    "reference_text": row["reference_text"],
                    "gt_translation": row["translated_text"],
                    "messages": [
                        {"role": "system", "content": row_translation_prompt},
                        {
                            "role": "user",
                            "content": f"Convert the following text, which has been encoded according to the provided scheme, back to English:\n\n{row['reference_text']}",
                        },
                        {"role": "assistant", "content": row["translated_text"]},
                    ],
                }
            )

        df_sft = pd.DataFrame(l_inputs)
        path = os.path.join("output", experiment_hash, "data", f"sft{suffix}.parquet")
        df_sft.to_parquet(path)

        print(f"Wrote {path}")

        n_tokens = df_sft["messages"].map(count_tokens_from_messages).sum()
        print(f"Got {n_tokens} tokens for {path}")


@ray.remote(num_cpus=1, num_gpus=4, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
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
        if len(row["reference_text"]) > 4000:
            n_skipped += 1
            continue

        row_translation_prompt = translation_prompt
        if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
            row_translation_prompt += "\n" + row["few_shot_examples"]

        l_inputs.append(
            {
                "reference_text": row["reference_text"],
                "gt_translation": row["translated_text"],
                "prompt": [
                    {"role": "system", "content": row_translation_prompt},
                    {
                        "role": "user",
                        "content": f"Convert the following text, which has been encoded according to the provided scheme, back to English:\n\n{row['reference_text']}",
                    },
                ],
            }
        )

    print(f"Skipped {n_skipped} rows because they were too long.")

    # Generate the outputs

    sampling_model = config["experiment"]["experiment_params"]["model"]
    assert "Qwen" in sampling_model, "RoPE scaling for Llama not yet implemented"
    model_size = int(re.search("([0-9]+)B", sampling_model).group(1))

    if config["experiment"]["experiment_params"].get("use_sft_model_for_sampling", False):
        sampling_model = f"output/{experiment_hash}/sft_model/last"
        print(f"Using SFT model {sampling_model} for translation instead...")

    llm = LLM(
        model=sampling_model,
        enforce_eager=True,
        gpu_memory_utilization=0.7,
        rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
        max_model_len=131072,
        tensor_parallel_size=4,
    )
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
    gt_logprobs = [[next(iter(l.values())) for l in logprob] for logprob in gt_logprobs]
    gt_logprob_toks = [[l.decoded_token for l in logprob] for logprob in gt_logprobs]
    gt_logprobs = [[l.logprob for l in logprob] for logprob in gt_logprobs]

    for i, gt_logprob in enumerate(gt_logprobs):
        l_inputs[i]["gt_logprobs"] = gt_logprob
        l_inputs[i]["gt_logprob_tokens"] = gt_logprob_toks[i]

    df_output = pd.DataFrame(l_inputs)
    df_output.to_parquet(os.path.join("output", experiment_hash, "data", "prompted_translation.parquet"))

    kill_vllm_process(llm)
