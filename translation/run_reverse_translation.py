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

    fn_encoding_scheme = get_encoding_scheme(
        config["experiment"]["experiment_params"]["encoding_scheme"], config
    )

    dataset_name = config["experiment"]["experiment_params"]["dataset"]
    if dataset_override:
        dataset_name = dataset_override
    dataset = get_dataset(dataset_name)

    experiment_hash = compute_experiment_hash(config)
    target_path = os.path.join(
        "output", experiment_hash, "data", "ground_truth_translation.parquet"
    )

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    async def gather_all(tasks):
        return await asyncio.gather(*tasks)

    ref_translation_cot = [None for _ in range(len(dataset))]

    translated_text = [fn_encoding_scheme(s) for s in dataset]
    if is_async_encoding_scheme(
        config["experiment"]["experiment_params"]["encoding_scheme"]
    ):
        translated_text = asyncio.run(gather_all(translated_text))
        ref_translation_cot = [t[1] for t in translated_text]
        translated_text = [t[0] for t in translated_text]

    # Note that translated is the reference input and the English is the translated target!
    df = pd.DataFrame(
        {
            "reference_text": translated_text,
            "translated_text": dataset,
            "ref_translation_cot": ref_translation_cot,
        }
    )

    if config["experiment"]["experiment_params"].get("validation_set_frac", 0):
        validation_set_frac = config["experiment"]["experiment_params"][
            "validation_set_frac"
        ]
        train_set_frac = 1.0 - validation_set_frac

        df_train = df.sample(frac=train_set_frac, random_state=42)
        df_valid = df[~df.index.isin(df_train.index)]

        train_path = os.path.join(
            "output", experiment_hash, "data", "ground_truth_translation_train.parquet"
        )
        df_train.to_parquet(train_path)
        df_valid.to_parquet(target_path)
    else:
        df.to_parquet(target_path)


def get_few_shot_examples(df, df_sample_group, config):
    n_few_shot_examples = config["experiment"]["experiment_params"].get(
        "n_few_shot_examples", 0
    )

    l_few_shot_examples = []

    for i, row in df.iterrows():
        df_sample = df_sample_group[
            df_sample_group["translated_text"] != row["translated_text"]
        ]
        df_sample = df_sample.sample(n=n_few_shot_examples, random_state=42)

        s = "\n"
        idx = 0
        for j, sample_row in df_sample.iterrows():
            idx += 1

            s += (
                f"Example {idx}. Input: {sample_row['reference_text']} Output: {sample_row['translated_text']}"
                + "\n"
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
        target_path = os.path.join(
            "output",
            experiment_hash,
            "data",
            f"ground_truth_translation{suffix}.parquet",
        )
        df = pd.read_parquet(target_path)

        df["len"] = df["translated_solution"].map(len)
        df_sample_group = df.sort_values("len")
        df_sample_group = df_sample_group[
            df_sample_group["translated_solution"].map(lambda x: "\\boxed{}" not in x)
        ]
        df_sample_group = df_sample_group.head(100)
        df = df.drop(columns=["len"])

        df["few_shot_examples"] = get_few_shot_examples(df, df_sample_group, config)
        df.to_parquet(target_path)


@ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 32)
def generate_sft_dataset(
    config,
    skip_too_long=True,
    reference_text_col="reference_text",
    translated_text_col="translated_text",
):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts import get_translation_prompt
    from translation.run_translation import count_tokens_from_messages

    experiment_hash = compute_experiment_hash(config)

    for suffix in ["", "_train"]:
        ground_truth_translation = pd.read_parquet(
            os.path.join(
                "output",
                experiment_hash,
                "data",
                f"ground_truth_translation{suffix}.parquet",
            )
        )

        # Build the prompt
        translation_prompt = get_translation_prompt(
            config["experiment"]["experiment_params"]["translation_prompt"]
        )

        n_skipped = 0

        l_inputs = []
        for i, row in ground_truth_translation.iterrows():
            if len(row[reference_text_col]) > 4000 and skip_too_long:
                n_skipped += 1
                continue

            row_translation_prompt = translation_prompt
            if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
                if (
                    reference_text_col != "reference_text"
                    or translated_text_col != "translated_text"
                ):
                    print(
                        "WARNING: reference_text_col or translated_text_col not default but asked for few shot examples. Please check your few shot examples are as expected."
                    )
                row_translation_prompt += "\n" + row["few_shot_examples"]

            l_inputs.append(
                {
                    "reference_text": row[reference_text_col],
                    "gt_translation": row[translated_text_col],
                    "messages": [
                        {"role": "system", "content": row_translation_prompt},
                        {
                            "role": "user",
                            "content": f"Convert the following text, which has been encoded according to the provided scheme, back to English:\n\n{row[reference_text_col]}",
                        },
                        {"role": "assistant", "content": row[translated_text_col]},
                    ],
                }
            )

        df_sft = pd.DataFrame(l_inputs)
        path = os.path.join("output", experiment_hash, "data", f"sft{suffix}.parquet")
        df_sft.to_parquet(path)

        print(f"Wrote {path}")

        n_tokens = df_sft["messages"].map(count_tokens_from_messages).sum()
        print(f"Got {n_tokens} tokens for {path}")


@ray.remote(
    num_cpus=1, num_gpus=2, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32
)
def generate_prompted_translation(
    config,
    skip_too_long=True,
    reference_text_col="reference_text",
    translated_text_col="translated_text",
    translation_prompt_override=None,
    model_path_override=None,
    save_path_override=None,
    sampling_temperature_override=None,
):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts import get_translation_prompt
    from utils.vllm import kill_vllm_process, get_assistant_turn_token_boundaries

    experiment_hash = compute_experiment_hash(config)

    ground_truth_translation = pd.read_parquet(
        os.path.join(
            "output", experiment_hash, "data", "ground_truth_translation.parquet"
        )
    )

    # Build the prompt
    translation_prompt = config["experiment"]["experiment_params"]["translation_prompt"]
    if translation_prompt_override is not None:
        translation_prompt = translation_prompt_override
    translation_prompt = get_translation_prompt(translation_prompt)

    n_skipped = 0

    l_inputs = []
    for i, row in ground_truth_translation.iterrows():
        if len(row[reference_text_col]) > 4000 and skip_too_long:
            n_skipped += 1
            continue

        row_translation_prompt = translation_prompt
        if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
            if (
                reference_text_col != "reference_text"
                or translated_text_col != "translated_text"
            ):
                print(
                    "WARNING: reference_text_col or translated_text_col not default but asked for few shot examples. Please check your few shot examples are as expected."
                )

            row_translation_prompt += "\n" + row["few_shot_examples"]

        l_inputs.append(
            {
                # note that reference text here is the encoded form.
                "reference_text": row[reference_text_col],
                "gt_translation": row[translated_text_col],
                "prompt": [
                    {"role": "system", "content": row_translation_prompt},
                    {
                        "role": "user",
                        "content": f"Convert the following text, which has been encoded according to the provided scheme, back to English:\n\n{row[reference_text_col]}",
                    },
                ],
            }
        )

    print(f"Skipped {n_skipped} rows because they were too long.")

    # Generate the outputs

    sampling_model = config["experiment"]["experiment_params"]["model"]
    assert "Qwen" in sampling_model, "RoPE scaling for Llama not yet implemented"
    model_size = int(re.search("([0-9]+)B", sampling_model).group(1))

    tokenizer = AutoTokenizer.from_pretrained(sampling_model)

    if config["experiment"]["experiment_params"].get(
        "use_sft_model_for_sampling", False
    ):
        sampling_model = f"output/{experiment_hash}/sft_model/last"
        print(f"Using SFT model {sampling_model} for translation instead...")

    if model_path_override is not None:
        sampling_model = model_path_override.replace("__HASH__", experiment_hash)
        print(f"Using model path override {sampling_model}")

    llm = LLM(
        model=sampling_model,
        enforce_eager=True,
        gpu_memory_utilization=0.7,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
        max_model_len=131072,
        tensor_parallel_size=2,
    )

    temperature = config["experiment"]["experiment_params"]["sampling_params"][
        "temperature"
    ]
    if sampling_temperature_override is not None:
        if type(sampling_temperature_override) is str:
            temperature = float(sampling_temperature_override)
        else:
            temperature = sampling_temperature_override

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=12000,
        n=config["experiment"]["experiment_params"]["sampling_params"]["n"],
    )

    outputs = llm.chat(
        [r["prompt"] for r in l_inputs], sampling_params=sampling_params, use_tqdm=True
    )

    l_input_token_lens = [len(o.prompt_token_ids) for o in outputs]
    for i, output in enumerate(outputs):
        l_inputs[i]["model_translations"] = [choice.text for choice in output.outputs]

    # Compute logprobs on GT for perplexity calculations
    logprobs_sampling_params = SamplingParams(
        temperature=config["experiment"]["experiment_params"]["sampling_params"][
            "temperature"
        ],
        max_tokens=1,
        logprobs=0,
        prompt_logprobs=1,
        n=1,
    )
    l_logprobs_prompts = []
    l_start_end = []

    for i, row in enumerate(l_inputs):
        prompt = [
            *row["prompt"],
            {
                "role": "assistant",
                "content": row["gt_translation"],
            },
        ]
        l_logprobs_prompts.append(prompt)
        l_start_end.append(get_assistant_turn_token_boundaries(prompt, tokenizer))

    logprobs = llm.chat(
        l_logprobs_prompts, sampling_params=logprobs_sampling_params, use_tqdm=True
    )
    gt_logprobs = [
        o.prompt_logprobs[l_start_end[i][0] : l_start_end[i][1]]
        for i, o in enumerate(logprobs)
    ]
    gt_logprobs = [[next(iter(l.values())) for l in logprob] for logprob in gt_logprobs]
    gt_logprob_toks = [[l.decoded_token for l in logprob] for logprob in gt_logprobs]
    gt_logprobs = [[l.logprob for l in logprob] for logprob in gt_logprobs]

    for i, gt_logprob in enumerate(gt_logprobs):
        l_inputs[i]["gt_logprobs"] = gt_logprob
        l_inputs[i]["gt_logprob_tokens"] = gt_logprob_toks[i]

    df_output = pd.DataFrame(l_inputs)

    save_path = os.path.join(
        "output", experiment_hash, "data", "prompted_translation.parquet"
    )
    if save_path_override is not None:
        save_path = save_path_override.replace("__HASH__", experiment_hash)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_output.to_parquet(save_path)

    kill_vllm_process(llm)


@ray.remote(num_cpus=1, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def generate_openai_prompted_translation(
    config,
    skip_too_long=True,
    reference_text_col="reference_text",
    translated_text_col="translated_text",
    translation_prompt_override=None,
    system_prompt_override=None,
    user_prompt_suffix_override=None,
    sampling_temperature_override=None,
    prompt_prefix_override=None,
    translation_extraction_tag=None,
):
    from openai import AsyncOpenAI
    from asyncio import Semaphore
    import asyncio
    from tqdm.asyncio import tqdm

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts import get_translation_prompt
    from env.openai import set_openai_key
    from env.anthropic import set_anthropic_key

    set_openai_key()
    set_anthropic_key()

    experiment_hash = compute_experiment_hash(config)

    ground_truth_translation = pd.read_parquet(
        os.path.join(
            "output", experiment_hash, "data", "ground_truth_translation.parquet"
        )
    )

    # Build the prompt
    translation_prompt = config["experiment"]["experiment_params"]["translation_prompt"]
    if translation_prompt_override is not None:
        translation_prompt = translation_prompt_override
    translation_prompt = get_translation_prompt(translation_prompt)

    n_skipped = 0

    l_inputs = []
    for i, row in ground_truth_translation.iterrows():
        if len(row[reference_text_col]) > 4000 and skip_too_long:
            n_skipped += 1
            continue

        row_translation_prompt = translation_prompt
        if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
            if (
                reference_text_col != "reference_text"
                or translated_text_col != "translated_text"
            ):
                print(
                    "WARNING: reference_text_col or translated_text_col not default but asked for few shot examples. Please check your few shot examples are as expected."
                )

            row_translation_prompt += "\n" + row["few_shot_examples"]

        if system_prompt_override is not None:
            row_translation_prompt = system_prompt_override

        user_prompt_suffix = ""
        if user_prompt_suffix_override is not None:
            user_prompt_suffix = user_prompt_suffix_override

        prefix = f"Do not output anything other than your conversion (do not think before outputting). {user_prompt_suffix}"
        if prompt_prefix_override is not None:
            prefix = prompt_prefix_override

        l_inputs.append(
            {
                # note that reference text here is the encoded form.
                "reference_text": row[reference_text_col],
                "gt_translation": row[translated_text_col],
                "prompt": [
                    {"role": "system", "content": row_translation_prompt},
                    {
                        "role": "user",
                        "content": f"{prefix}Convert the following text, which has been encoded according to the provided scheme, back to English:\n\n{row[reference_text_col]}",
                    },
                ],
            }
        )

    print(f"Skipped {n_skipped} rows because they were too long.")

    # Generate the outputs
    base_url = config["experiment"]["experiment_params"]["base_url"]
    model_name = config["experiment"]["experiment_params"]["model"]
    temperature = config["experiment"]["experiment_params"]["sampling_params"][
        "temperature"
    ]
    if sampling_temperature_override is not None:
        if type(sampling_temperature_override) is str:
            temperature = float(sampling_temperature_override)
        else:
            temperature = sampling_temperature_override

    api_key = (
        os.environ["ANTHROPIC_API_KEY"]
        if "claude" in model_name
        else os.environ["OPENAI_API_KEY"]
    )

    d_additional_kwargs = {}

    if config["experiment"]["experiment_params"].get(
        "use_api_sft_model_for_sampling", False
    ):
        model_json_path = os.path.join(
            "output", experiment_hash, "data", "sft_model_meta.json"
        )
        with open(model_json_path, "r") as fp:
            d_model_json = json.load(fp)
        model_name = d_model_json["fine_tuned_model"]
        print(f"Using FT model {model_name}")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    rate_limit = Semaphore(100)

    async def run_chat(conversation):
        max_tokens = 12000
        if (
            model_name.startswith("claude-3-haiku")
            or model_name.startswith("claude-3-opus")
            or model_name.startswith("claude-3-5-haiku")
        ):
            max_tokens = 4096
        if model_name.startswith("claude-3-5-sonnet-20241022"):
            max_tokens = 8192

        for i in range(20000):
            try:
                async with rate_limit:
                    resp = await client.chat.completions.create(
                        model=model_name,
                        messages=conversation,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        **d_additional_kwargs,
                    )

                    ret = resp.choices[0].message.content
                    print(conversation)
                    print(ret)

                    return ret
            except Exception as e:
                print(e)
                await asyncio.sleep(15)

        print(f"{conversation} \n ran out of retries in limit!")
        raise Exception(f"Ran out of retries! {conversation}")

    async def gather_all(tasks):
        return await tqdm.gather(*tasks)

    l_responses = []
    for i in range(len(l_inputs)):
        l_responses.append(run_chat(l_inputs[i]["prompt"]))
    l_responses = asyncio.run(gather_all(l_responses))

    for i in range(len(l_responses)):
        if translation_extraction_tag is not None:
            result = re.search(
                f"<{translation_extraction_tag}>(.*?)</{translation_extraction_tag}>",
                l_responses[i],
                re.DOTALL,
            )
            if result:
                translation = result.group(1)
            else:
                translation = l_responses[i]

            l_inputs[i]["model_translations"] = [translation]
            l_inputs[i]["raw_model_translations"] = [l_responses[i]]
        else:
            l_inputs[i]["model_translations"] = [l_responses[i]]

        l_inputs[i]["gt_logprobs"] = [np.nan]
        l_inputs[i]["gt_logprob_tokens"] = ["a"]

    df_output = pd.DataFrame(l_inputs)
    df_output.to_parquet(
        os.path.join("output", experiment_hash, "data", "prompted_translation.parquet")
    )


@ray.remote(num_cpus=1, memory=32 * 1024 * 1024 * 1024)
def compute_openai_validation_loss(
    config,
    validation_set_name=None,
    skip_too_long=True,
    reference_text_col="reference_text",
    translated_text_col="translated_text",
    validation_parquet_override=None,
    use_base_instruct_model=False,
    override_source_validation_data_template=None,
):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from sft.sft_runner import (
        get_valid_loss_for_openai_job,
        openai_sft_model,
        write_test_sft_data_for_extracting_validation_loss,
    )
    from prompts import get_translation_prompt

    assert validation_set_name is not None

    experiment_hash = compute_experiment_hash(config)

    hash_dir = os.path.join("output", experiment_hash)
    train_parquet_path = os.path.join(
        hash_dir, "data", f"validation_{validation_set_name}_train.parquet"
    )
    train_json_path = os.path.join(
        hash_dir, "data", f"validation_{validation_set_name}_train.jsonl"
    )
    valid_parquet_path = os.path.join(
        hash_dir, "data", f"validation_{validation_set_name}_valid.parquet"
    )
    valid_json_path = os.path.join(
        hash_dir, "data", f"validation_{validation_set_name}_valid.jsonl"
    )
    model_json_path = os.path.join(
        hash_dir, "data", f"validation_{validation_set_name}_meta.json"
    )

    # Build the validation Parquet file from the reference GT file
    if not override_source_validation_data_template:
        ground_truth_path = os.path.join(
            "output", experiment_hash, "data", "ground_truth_translation.parquet"
        )
        if validation_parquet_override:
            ground_truth_path = validation_parquet_override
        ground_truth_translation = pd.read_parquet(ground_truth_path)

        # Build the prompt
        translation_prompt = get_translation_prompt(
            config["experiment"]["experiment_params"]["translation_prompt"]
        )

        n_skipped = 0

        l_inputs = []
        for i, row in ground_truth_translation.iterrows():
            if len(row[reference_text_col]) > 4000 and skip_too_long:
                n_skipped += 1
                continue

            row_translation_prompt = translation_prompt
            if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
                if (
                    reference_text_col != "reference_text"
                    or translated_text_col != "translated_text"
                ):
                    raise ValueError(
                        "reference_text_col or translated_text_col not default but asked for few shot examples. This is not yet implemented."
                    )

                row_translation_prompt += "\n" + row["few_shot_examples"]

            l_inputs.append(
                {
                    # note that reference text here is the encoded form.
                    "messages": [
                        {"role": "system", "content": row_translation_prompt},
                        {
                            "role": "user",
                            "content": f"Convert the following text, which has been encoded according to the provided scheme, back to English:\n\n{row[reference_text_col]}",
                        },
                        {"role": "assistant", "content": row[translated_text_col]},
                    ],
                }
            )

        print(f"Skipped {n_skipped} rows because they were too long.")

        df_inputs = pd.DataFrame(l_inputs)
        df_inputs.to_parquet(valid_parquet_path)

        print(f"Wrote {valid_parquet_path}")
    else:
        override_source_validation_data_template = (
            override_source_validation_data_template.replace(
                "__HASH__", experiment_hash
            )
        )
        valid_parquet_path = override_source_validation_data_template

    write_test_sft_data_for_extracting_validation_loss(train_parquet_path)

    ref_sft_model_meta_path = os.path.join(hash_dir, "data", "sft_model_meta.json")
    with open(ref_sft_model_meta_path, "r") as fp:
        d_ref_sft_model_meta = json.load(fp)
    if use_base_instruct_model:
        model_override = None
    else:
        model_override = d_ref_sft_model_meta["fine_tuned_model"]

    # Kick off hacked FT run
    ray.get(
        openai_sft_model.remote(
            config,
            train_parquet_path,
            train_json_path,
            model_json_path,
            valid_parquet_path,
            valid_json_path,
            finetuning_parameters={
                "batch_size": 10,
                "learning_rate_multiplier": 0.0001,
                "n_epochs": 1,
            },
            model_override=model_override,
        )
    )

    valid_loss = get_valid_loss_for_openai_job(model_json_path)
    with open(model_json_path, "r") as fp:
        d_model_meta = json.load(fp)

    d_model_meta["valid_loss"] = valid_loss
    with open(model_json_path, "w") as fp:
        json.dump(d_model_meta, fp)
