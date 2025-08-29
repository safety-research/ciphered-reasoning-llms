import numpy as np
import os
import sys
import ray
import pandas as pd
import tiktoken
import json
import re
import asyncio
from openai import AsyncOpenAI


encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens_from_messages(messages):
    total_tokens = 0
    for msg in messages:
        # Count role and content separately
        total_tokens += len(encoding.encode(msg.get("role", "")))
        total_tokens += len(encoding.encode(msg.get("content", "")))
    return total_tokens


@ray.remote(num_cpus=1)
def generate_ground_truth_translation(config, dataset_override=None, validation_set_frac_override=None, write_as_train_file=False):
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

    translated_solution = [fn_encoding_scheme(r["solution"]) for r in dataset]
    if is_async_encoding_scheme(config["experiment"]["experiment_params"]["encoding_scheme"]):
        translated_solution = asyncio.run(gather_all(translated_solution))

    df = pd.DataFrame(
        {
            "reference_problem": [r["problem"] for r in dataset],
            "reference_solution": [r["solution"] for r in dataset],
            "translated_solution": [
                sol + f"\n\\boxed{{{r['answer']}}}" for sol, r in zip(translated_solution, dataset)
            ],
            "answer": [r["answer"] for r in dataset],
        }
    )

    validation_set_frac = config["experiment"]["experiment_params"].get("validation_set_frac", 0)
    if validation_set_frac_override is not None:
        validation_set_frac = validation_set_frac_override

    train_path = os.path.join("output", experiment_hash, "data", "ground_truth_translation_train.parquet")
    if validation_set_frac:
        train_set_frac = 1.0 - validation_set_frac

        df_train = df.sample(frac=train_set_frac, random_state=42)
        df_valid = df[~df.index.isin(df_train.index)]
        
        df_train.to_parquet(train_path)
        df_valid.to_parquet(target_path)
    elif write_as_train_file:
        df.to_parquet(train_path)
    else:
        df.to_parquet(target_path)


def get_few_shot_examples(df, df_sample_group, config):
    n_few_shot_examples = config["experiment"]["experiment_params"].get("n_few_shot_examples", 0)

    l_few_shot_examples = []

    for i, row in df.iterrows():
        df_sample = df_sample_group[df_sample_group["translated_solution"] != row["translated_solution"]]
        df_sample = df_sample.sample(n=n_few_shot_examples, random_state=42)

        s = "\n"
        for j, sample_row in df_sample.iterrows():
            s += (
                f"Example {j + 1}. Question: {sample_row['reference_problem']} Output: {sample_row['translated_solution']}"
                + "\n"
            )

        l_few_shot_examples.append(s)

    return l_few_shot_examples


@ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 32)
def generate_fewshot_prompt(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from speaking.encoded_cot_runner import get_few_shot_examples

    experiment_hash = compute_experiment_hash(config)

    l_suffixes = [""]
    if config["experiment"]["experiment_params"].get("validation_set_frac", 0):
        l_suffixes.append("_train")

    for suffix in l_suffixes:
        target_path = os.path.join("output", experiment_hash, "data", f"ground_truth_translation{suffix}.parquet")
        df = pd.read_parquet(target_path)

        df["len"] = df["translated_solution"].map(len)
        df_sample_group = df.sort_values("len").head(100)
        df = df.drop(columns=["len"])

        df["few_shot_examples"] = get_few_shot_examples(df, df_sample_group, config)
        df.to_parquet(target_path)


@ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 32)
def generate_sft_dataset(config):
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

        l_inputs = []
        for i, row in ground_truth_translation.iterrows():
            row_translation_prompt = translation_prompt
            if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
                row_translation_prompt += "\n" + row["few_shot_examples"]

            l_inputs.append(
                {
                    "messages": [
                        {"role": "system", "content": row_translation_prompt},
                        {
                            "role": "user",
                            "content": f"{row['reference_problem']}\nThink step by step, making sure that your thinking is encoded according to the instructions. Then, provide your final answer in \\boxed{{}} without any encoding.",
                        },
                        {"role": "assistant", "content": row["translated_solution"]},
                    ],
                }
            )

        df_sft = pd.DataFrame(l_inputs)
        path = os.path.join("output", experiment_hash, "data", f"sft{suffix}.parquet")
        df_sft.to_parquet(path)

        print(f"Wrote {path}")

        n_tokens = df_sft["messages"].map(count_tokens_from_messages).sum()
        print(f"Got {n_tokens} tokens for {path}")


@ray.remote(num_cpus=1, num_gpus=2, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
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

    l_inputs = []
    for i, row in ground_truth_translation.iterrows():

        row_translation_prompt = translation_prompt
        if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
            row_translation_prompt += "\n" + row["few_shot_examples"]

        l_inputs.append(
            {
                "answer": row["answer"],
                "reference_problem": row["reference_problem"],
                "reference_solution": row["reference_solution"],
                "prompt": [
                    {"role": "system", "content": row_translation_prompt},
                    {
                        "role": "user",
                        "content": f"{row['reference_problem']}\nThink step by step, making sure that your thinking is encoded according to the instructions. Then, provide your final answer in \\boxed{{}} without any encoding.",
                    },
                ],
            }
        )

    # Generate the outputs

    sampling_model = config["experiment"]["experiment_params"]["model"]
    assert "Qwen" in sampling_model, "RoPE scaling for Llama not yet implemented"
    model_size = int(re.search("([0-9]+)B", sampling_model).group(1))

    if config["experiment"]["experiment_params"].get("use_sft_model_for_sampling", False):
        sampling_model = f"output/{experiment_hash}/sft_model/last"
        print(f"Using SFT model {sampling_model} for generation instead...")

    llm = LLM(
        model=sampling_model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
        max_model_len=131072,
        tensor_parallel_size=2,
    )

    extra_sampling_kwargs = {}
    if config["experiment"]["experiment_params"]["encoding_scheme"] == "speaking_zero_shot":
        from vllm.sampling_params import GuidedDecodingParams
        extra_sampling_kwargs['guided_decoding'] = GuidedDecodingParams(regex=r"\\boxed\{.+\}")
    sampling_params = SamplingParams(
        temperature=config["experiment"]["experiment_params"]["sampling_params"]["temperature"],
        max_tokens=12000,
        n=config["experiment"]["experiment_params"]["sampling_params"]["n"],
        **extra_sampling_kwargs
    )

    outputs = llm.chat([r["prompt"] for r in l_inputs], sampling_params=sampling_params, use_tqdm=True)

    l_input_token_lens = [len(o.prompt_token_ids) for o in outputs]
    for i, output in enumerate(outputs):
        l_inputs[i]["model_cot"] = [choice.text for choice in output.outputs]

    df_output = pd.DataFrame(l_inputs)
    df_output.to_parquet(os.path.join("output", experiment_hash, "data", "prompted_cot.parquet"))

    kill_vllm_process(llm)


@ray.remote(num_cpus=4, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def generate_openai_prompted_translation(config):
    from openai import AsyncOpenAI
    from asyncio import Semaphore
    import asyncio
    from tqdm.asyncio import tqdm

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts import get_translation_prompt
    from env.openai import set_openai_key

    set_openai_key()

    experiment_hash = compute_experiment_hash(config)

    ground_truth_translation = pd.read_parquet(
        os.path.join("output", experiment_hash, "data", "ground_truth_translation.parquet")
    )

    # Build the prompt
    translation_prompt_type = config["experiment"]["experiment_params"]["translation_prompt"] 
    translation_prompt = get_translation_prompt(translation_prompt_type)

    if translation_prompt_type == "speaking_zero_shot":
        # prefill answer
        l_prefill = [{
            "role": "assistant",
            "content": r"\boxed"
        }]
    else:
        l_prefill = []

    l_inputs = []
    for i, row in ground_truth_translation.iterrows():

        row_translation_prompt = translation_prompt
        if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
            row_translation_prompt += "\n" + row["few_shot_examples"]

        l_inputs.append(
            {
                "answer": row["answer"],
                "reference_problem": row["reference_problem"],
                "reference_solution": row["reference_solution"],
                "prompt": [
                    {"role": "system", "content": row_translation_prompt},
                    {
                        "role": "user",
                        "content": f"{row['reference_problem']}\nThink step by step, making sure that your thinking is encoded according to the instructions. Then, provide your final answer in \\boxed{{}} without any encoding.",
                    },
                ] + l_prefill,
            }
        )

    # Generate the outputs
    base_url = config["experiment"]["experiment_params"]["base_url"]
    model_name = config["experiment"]["experiment_params"]["model"]
    temperature = config["experiment"]["experiment_params"]["sampling_params"]["temperature"]
    api_key = os.environ['ANTHROPIC_API_KEY'] if 'claude' in model_name else os.environ["OPENAI_API_KEY"]

    if config["experiment"]["experiment_params"].get("use_api_sft_model_for_sampling", False):
        model_json_path = os.path.join("output", experiment_hash, "data", "sft_model_meta.json")
        with open(model_json_path, "r") as fp:
            d_model_json = json.load(fp)
        model_name = d_model_json["fine_tuned_model"]
        print(f"Using FT model {model_name}")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
 
    rate_limit = Semaphore(30)
    async def run_chat(conversation):
        max_tokens = 12000
        if model_name.startswith("claude-3-haiku") or model_name.startswith("claude-3-opus") or model_name.startswith("claude-3-5-haiku"):
            max_tokens = 4096
        if model_name.startswith("claude-3-5-sonnet-20241022"):
            max_tokens = 8192

        for i in range(200):
            try:
                async with rate_limit:
                    resp = await client.chat.completions.create(
                        model=model_name,
                        messages=conversation,
                        temperature=temperature,
                        max_completion_tokens=max_tokens
                    )

                    ret = resp.choices[0].message.content
                    print(conversation)
                    print(ret)

                    if translation_prompt_type == "speaking_zero_shot":
                        ret = fr"\boxed{ret}"

                    return ret
            except Exception as e:
                print(e)
                await asyncio.sleep(3)

        print(f"{conversation} \n ran out of retries in limit!")
        return "Error occurred."

    async def gather_all(tasks):
        return await tqdm.gather(*tasks)

    l_responses = []
    for i in range(len(l_inputs)):
        l_responses.append(run_chat(l_inputs[i]["prompt"]))
    l_responses = asyncio.run(gather_all(l_responses))

    for i in range(len(l_responses)):
        l_inputs[i]["model_cot"] = [l_responses[i]]

    df_output = pd.DataFrame(l_inputs)
    df_output.to_parquet(os.path.join("output", experiment_hash, "data", "prompted_cot.parquet"))


@ray.remote(num_cpus=1, num_gpus=2, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def judge_cot_style_adherence(config):
    from vllm import LLM, SamplingParams

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts.translation.judge import followed_encoding_style_judge
    from prompts import get_translation_prompt
    from utils.vllm import kill_vllm_process

    experiment_hash = compute_experiment_hash(config)

    generated_cot_path = os.path.join("output", experiment_hash, "data", "prompted_cot.parquet")
    df_generated_cot = pd.read_parquet(generated_cot_path)

    sft_ref_path = os.path.join("output", experiment_hash, "data", "sft.parquet")
    df_sft = pd.read_parquet(sft_ref_path)

    # Ask LLM for inference
    llm = LLM(
        model="Qwen/Qwen3-32B-FP8",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
        max_model_len=131072,
        tensor_parallel_size=2
    )

    translation_prompt_type = config["experiment"]["experiment_params"]["translation_prompt"] 
    translation_prompt = get_translation_prompt(translation_prompt_type)

    followed_encoding_style_judge = followed_encoding_style_judge + f"\n<instruction>\n{translation_prompt}\n</instruction>"

    l_judge_prompts = []
    for (_, generated_cot_row), (_, sft_row) in zip(df_generated_cot.iterrows(), df_sft.iterrows()):
        sft_reference = sft_row['messages'][-1]['content']

        for cot in generated_cot_row['model_cot']:
            l_judge_prompts.append([{"role": "system", "content": "/no_think"}, {"role": "user", "content": followed_encoding_style_judge + f"\n<text>{cot}</text>\n<reference_text>{sft_reference}</reference_text>"}])

    judge_sampling_params = SamplingParams(max_tokens=1024)
    outputs = llm.chat(l_judge_prompts, sampling_params=judge_sampling_params, use_tqdm=True)
    outputs_idx = 0
    l_judge_scores = []

    for cots in df_generated_cot['model_cot']:
        l_instance_scores = []
        for cot in cots:
            text = outputs[outputs_idx].outputs[0].text
            outputs_idx += 1

            search_result = re.search("<answer>(.*?)</answer>", text)
            if search_result:
                l_instance_scores.append(1.0 if search_result.group(1) == "Yes" else 0.0)
            else:
                l_instance_scores.append(0.0)

        l_judge_scores.append(l_instance_scores)

    df_generated_cot["followed_encoding_style"] = l_judge_scores

    df_generated_cot.to_parquet(generated_cot_path)

    kill_vllm_process(llm)


@ray.remote(num_cpus=1, num_gpus=2, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def judge_cot_encoding_English_coherence(config):
    from vllm import LLM, SamplingParams

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts.translation.judge import coherent_english_judge
    from encoding_schemes import get_inverse_encoding_scheme, is_async_encoding_scheme
    from utils.vllm import kill_vllm_process

    experiment_hash = compute_experiment_hash(config)

    target_path = os.path.join("output", experiment_hash, "data", "prompted_cot.parquet")

    df = pd.read_parquet(target_path)

    fn_inverse_encoding_scheme = get_inverse_encoding_scheme(
        config["experiment"]["experiment_params"]["encoding_scheme"], config
    )

    async def gather_all(tasks):
        return await asyncio.gather(*tasks)

    l_inverted_cot = [[fn_inverse_encoding_scheme(cot) for cot in cots] for cots in df["model_cot"]]
    if is_async_encoding_scheme(config["experiment"]["experiment_params"]["encoding_scheme"]):
        l_inverted_cot = [gather_all(cots) for cots in l_inverted_cot]
        l_inverted_cot = asyncio.run(gather_all(l_inverted_cot))

    # Ask LLM for inference
    llm = LLM(
        model="Qwen/Qwen3-32B-FP8",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
        max_model_len=131072,
        tensor_parallel_size=2
    )

    l_judge_prompts = []
    for cots in l_inverted_cot:
        for cot in cots:
            l_judge_prompts.append([{"role": "system", "content": "/no_think"}, {"role": "user", "content": coherent_english_judge + f"\n<text>{cot}</text>"}])

    judge_sampling_params = SamplingParams(max_tokens=1024)
    outputs = llm.chat(l_judge_prompts, sampling_params=judge_sampling_params, use_tqdm=True)
    outputs_idx = 0
    l_judge_scores = []

    for cots in l_inverted_cot:
        l_instance_scores = []
        for cot in cots:
            text = outputs[outputs_idx].outputs[0].text
            outputs_idx += 1

            search_result = re.search("<answer>(.*?)</answer>", text)
            if search_result:
                l_instance_scores.append(1.0 if search_result.group(1) == "Yes" else 0.0)
            else:
                l_instance_scores.append(0.0)

        l_judge_scores.append(l_instance_scores)

    df["english_coherence_scores"] = l_judge_scores
    df["decoded_cot"] = l_inverted_cot

    df.to_parquet(target_path)

    kill_vllm_process(llm)



@ray.remote(num_cpus=1, num_gpus=2, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def judge_math_solving_content(config):
    from vllm import LLM, SamplingParams

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts.translation.judge import doing_math_judge
    from utils.vllm import kill_vllm_process

    experiment_hash = compute_experiment_hash(config)

    generated_cot_path = os.path.join("output", experiment_hash, "data", "prompted_cot.parquet")
    df_generated_cot = pd.read_parquet(generated_cot_path)

    sft_ref_path = os.path.join("output", experiment_hash, "data", "sft.parquet")
    df_sft = pd.read_parquet(sft_ref_path)

    # Ask LLM for inference
    llm = LLM(
        model="Qwen/Qwen3-32B-FP8",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
        max_model_len=131072,
        tensor_parallel_size=2
    )

    l_judge_prompts = []
    for _, generated_cot_row in df_generated_cot.iterrows():
        for cot in generated_cot_row['model_cot']:
            l_judge_prompts.append([{"role": "system", "content": "/no_think"}, {"role": "user", "content": doing_math_judge + f"\n<text>{cot}</text>"}])

    judge_sampling_params = SamplingParams(max_tokens=1024)
    outputs = llm.chat(l_judge_prompts, sampling_params=judge_sampling_params, use_tqdm=True)
    outputs_idx = 0
    l_judge_scores = []

    for cots in df_generated_cot['model_cot']:
        l_instance_scores = []
        for cot in cots:
            text = outputs[outputs_idx].outputs[0].text
            outputs_idx += 1

            search_result = re.search("<answer>(.*?)</answer>", text)
            if search_result:
                l_instance_scores.append(1.0 if search_result.group(1) == "Yes" else 0.0)
            else:
                l_instance_scores.append(0.0)

        l_judge_scores.append(l_instance_scores)

    df_generated_cot["contains_math_solving"] = l_judge_scores

    df_generated_cot.to_parquet(generated_cot_path)

    kill_vllm_process(llm)


def ensure_fireworks_deployment(config):
    pass


def tear_down_fireworks_deployment(config):
    pass