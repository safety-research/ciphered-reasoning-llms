import numpy as np
import os
import sys
import ray
import pandas as pd
import tiktoken
import json
import re
import asyncio
from asyncio import Semaphore
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

    ref_translation_cot = [None for _ in range(len(dataset))]

    translated_solution = [fn_encoding_scheme(r["solution"]) for r in dataset]
    if is_async_encoding_scheme(config["experiment"]["experiment_params"]["encoding_scheme"]):
        translated_solution = asyncio.run(gather_all(translated_solution))

        ref_translation_cot = [t[1] for t in translated_solution]
        translated_solution = [t[0] for t in translated_solution]

    df = pd.DataFrame(
        {
            "reference_problem": [r["problem"] for r in dataset],
            "reference_solution": [r["solution"] for r in dataset],
            "translated_solution": [
                sol + f"\n\\boxed{{{r['answer']}}}" for sol, r in zip(translated_solution, dataset)
            ],
            "raw_translated_cot": translated_solution,
            "answer": [r["answer"] for r in dataset],
            "ref_translation_cot": ref_translation_cot
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

        if len(df_sample) == 0:
            print("!!!!! no few shot examples found !!!!!")
            print(df_sample_group['translated_solution'].head())
            print(row['translated_solution'])

        df_sample = df_sample.sample(n=n_few_shot_examples, random_state=42)

        s = "\n"
        for j, sample_row in df_sample.iterrows():
            s += (
                f"Example {j + 1}. Normal text: {sample_row['reference_solution']} Encoded text: {sample_row['translated_solution']}"
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
        df_sample_group = df.sort_values("len")
        df_sample_group = df_sample_group[df_sample_group['translated_solution'].map(lambda x: '\\boxed{}' not in x)]
        df_sample_group = df_sample_group.head(100)
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
                "translated_solution": row["translated_solution"],
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
                    "content": row["translated_solution"],
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
    from env.anthropic import set_anthropic_key

    set_openai_key()
    set_anthropic_key()

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

    d_additional_kwargs = {}
    if "gpt-5" in model_name:
        d_additional_kwargs["service_tier"] = "flex"

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

        for i in range(20000):
            try:
                async with rate_limit:
                    resp = await client.chat.completions.create(
                        model=model_name,
                        messages=conversation,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        **d_additional_kwargs
                    )

                    ret = resp.choices[0].message.content
                    print(conversation)
                    print(ret)

                    if translation_prompt_type == "speaking_zero_shot":
                        ret = fr"\boxed{ret}"

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
        l_inputs[i]["model_cot"] = [l_responses[i]]

        l_inputs[i]["gt_logprobs"] = [np.nan]
        l_inputs[i]["gt_logprob_tokens"] = ["a"]

    df_output = pd.DataFrame(l_inputs)
    df_output.to_parquet(os.path.join("output", experiment_hash, "data", "prompted_cot.parquet"))



def judge_cot_style_adherence_deterministically(config):
    from orchestration.experiment_meta_saver import compute_experiment_hash
    from encoding_schemes import get_deterministic_adherence_fn
    
    fn_adherence_evaluator = get_deterministic_adherence_fn(config["experiment"]["experiment_params"]["encoding_scheme"], config)

    experiment_hash = compute_experiment_hash(config)

    generated_cot_path = os.path.join("output", experiment_hash, "data", "prompted_cot.parquet")
    df_generated_cot = pd.read_parquet(generated_cot_path)

    l_judge_scores = []
    for cots in df_generated_cot['model_cot']:
        l_instance_scores = []
        for cot in cots:
            l_instance_scores.append(1.0 if fn_adherence_evaluator(cot) else 0.0)

        l_judge_scores.append(l_instance_scores)

    df_generated_cot["followed_encoding_style"] = l_judge_scores

    df_generated_cot.to_parquet(generated_cot_path)


@ray.remote(num_cpus=1, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def judge_cot_style_adherence(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from encoding_schemes import get_deterministic_adherence_fn
    from speaking.encoded_cot_runner import judge_cot_style_adherence_deterministically

    if get_deterministic_adherence_fn(config["experiment"]["experiment_params"]["encoding_scheme"], config) is not None:
        judge_cot_style_adherence_deterministically(config)
        return

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts.translation.judge import followed_encoding_style_judge
    from prompts import get_translation_prompt

    experiment_hash = compute_experiment_hash(config)

    generated_cot_path = os.path.join("output", experiment_hash, "data", "prompted_cot.parquet")
    df_generated_cot = pd.read_parquet(generated_cot_path)

    sft_ref_path = os.path.join("output", experiment_hash, "data", "sft.parquet")
    df_sft = pd.read_parquet(sft_ref_path)

    translation_prompt_type = config["experiment"]["experiment_params"]["translation_prompt"] 
    translation_prompt = get_translation_prompt(translation_prompt_type)

    followed_encoding_style_judge = followed_encoding_style_judge + f"\n<instruction>\n{translation_prompt}\n</instruction>"

    l_judge_prompts = []
    for (_, generated_cot_row), (_, sft_row) in zip(df_generated_cot.iterrows(), df_sft.iterrows()):
        sft_row = df_sft.iloc[0]

        sft_reference = sft_row['messages'][-1]['content']

        for cot in generated_cot_row['model_cot']:
            l_judge_prompts.append([{"role": "user", "content": followed_encoding_style_judge + f"\n<text>{cot}</text>\n<reference_text>{sft_reference}</reference_text>"}])

    api_key = os.environ['ANTHROPIC_API_KEY']

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.anthropic.com/v1/",
    )

    async def gather_all(tasks):
        return await asyncio.gather(*tasks)
 
    rate_limit = Semaphore(30)
    async def run_chat(conversation):
        max_tokens = 12000

        for i in range(200):
            try:
                async with rate_limit:
                    resp = await client.chat.completions.create(
                        model="claude-sonnet-4-20250514",
                        messages=conversation,
                        temperature=0.0,
                        max_completion_tokens=max_tokens
                    )

                    ret = resp.choices[0].message.content
                    print(conversation)
                    print(ret)

                    return ret
            except Exception as e:
                print(e)
                await asyncio.sleep(3)

        print(f"{conversation} \n ran out of retries in limit!")
        raise Exception(f"Ran out of retries! {conversation}")

    l_responses = []
    for i in range(len(l_judge_prompts)):
        l_responses.append(run_chat(l_judge_prompts[i]))
    l_responses = asyncio.run(gather_all(l_responses))

    outputs_idx = 0
    l_judge_scores = []

    for cots in df_generated_cot['model_cot']:
        l_instance_scores = []
        for cot in cots:
            text = l_responses[outputs_idx]
            outputs_idx += 1

            search_result = re.search("<answer>(.*?)</answer>", text)
            if search_result:
                l_instance_scores.append(1.0 if search_result.group(1) == "Yes" else 0.0)
            else:
                l_instance_scores.append(0.0)

        l_judge_scores.append(l_instance_scores)

    df_generated_cot["followed_encoding_style"] = l_judge_scores

    df_generated_cot.to_parquet(generated_cot_path)


@ray.remote(num_cpus=1, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def judge_cot_encoding_English_coherence(config):
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

    l_judge_prompts = []
    for cots in l_inverted_cot:
        for cot in cots:
            l_judge_prompts.append([{"role": "user", "content": coherent_english_judge + f"\n<text>{cot}</text>"}])

    api_key = os.environ['ANTHROPIC_API_KEY']

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.anthropic.com/v1/",
    )
 
    rate_limit = Semaphore(30)
    async def run_chat(conversation):
        max_tokens = 12000

        for i in range(200):
            try:
                async with rate_limit:
                    resp = await client.chat.completions.create(
                        model="claude-sonnet-4-20250514",
                        messages=conversation,
                        temperature=0.0,
                        max_completion_tokens=max_tokens
                    )

                    ret = resp.choices[0].message.content
                    print(conversation)
                    print(ret)

                    return ret
            except Exception as e:
                print(e)
                await asyncio.sleep(3)

        print(f"{conversation} \n ran out of retries in limit!")
        raise Exception(f"Ran out of retries! {conversation}")

    l_responses = []
    for i in range(len(l_judge_prompts)):
        l_responses.append(run_chat(l_judge_prompts[i]))
    l_responses = asyncio.run(gather_all(l_responses))

    outputs_idx = 0
    l_judge_scores = []

    for cots in l_inverted_cot:
        l_instance_scores = []
        for cot in cots:
            text = l_responses[outputs_idx]
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



@ray.remote(num_cpus=4, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def generate_together_prompted_translation(config):
    from together import AsyncTogether
    from asyncio import Semaphore
    import asyncio
    from tqdm.asyncio import tqdm
    from transformers import AutoTokenizer

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts import get_translation_prompt

    experiment_hash = compute_experiment_hash(config)
    data_dir = os.path.join("output", experiment_hash, "data")

    # Read deployment info to get model ID
    deployment_info_path = os.path.join(data_dir, "deployment_info.json")
    if not os.path.exists(deployment_info_path):
        raise FileNotFoundError(f"Deployment info not found at {deployment_info_path}. Please ensure deployment is created first.")
    
    with open(deployment_info_path, "r") as f:
        deployment_info = json.load(f)
    
    deployment_model_id = deployment_info.get("deployment_model_path", deployment_info.get("deployment_id"))
    if not deployment_model_id:
        raise ValueError("No deployment_model_path or deployment_id found in deployment_info.json")
    
    print(f"Using deployment model: {deployment_model_id}")
    
    # Load tokenizer for the model
    base_model_name = config["experiment"]["experiment_params"]["model"]
    print(f"Loading tokenizer for model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    ground_truth_translation = pd.read_parquet(
        os.path.join(data_dir, "ground_truth_translation.parquet")
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
                "translated_solution": row["translated_solution"],
                "prompt": [
                    {"role": "system", "content": row_translation_prompt},
                    {
                        "role": "user",
                        "content": f"{row['reference_problem']}\nThink step by step, making sure that your thinking is encoded according to the instructions. Then, provide your final answer in \\boxed{{}} without any encoding.",
                    },
                ] + l_prefill,
            }
        )

    # Initialize Together client
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    client = AsyncTogether(api_key=api_key)

    temperature = config["experiment"]["experiment_params"]["sampling_params"]["temperature"]
    n_samples = config["experiment"]["experiment_params"]["sampling_params"].get("n", 1)
    
    # Generate rollouts
    print(f"Generating {n_samples} rollout(s) for {len(l_inputs)} prompts...")
    
    rate_limit = Semaphore(30)  # Together AI has lower rate limits than OpenAI
    async def run_chat(conversation, include_logprobs=False):
        max_tokens = 12000
        
        for retry in range(20):
            try:
                async with rate_limit:
                    params = {
                        "model": deployment_model_id,
                        "messages": conversation,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    
                    if include_logprobs:
                        params["logprobs"] = 1
                        params["echo"] = True  # Include prompt tokens in logprobs
                    
                    resp = await client.chat.completions.create(**params)
                    
                    ret = resp.choices[0].message.content
                    print(f"Prompt: {conversation[-1]['content'][:100]}...")
                    print(f"Response: {ret[:200]}...")
                    
                    if translation_prompt_type == "speaking_zero_shot":
                        ret = fr"\boxed{ret}"
                    
                    if include_logprobs and hasattr(resp.choices[0], 'logprobs'):
                        return ret, resp.choices[0].logprobs
                    else:
                        return ret
                        
            except Exception as e:
                print(f"Error on retry {retry}: {e}")
                await asyncio.sleep(min(2 ** retry, 60))
        
        raise Exception(f"Ran out of retries for conversation: {conversation}")

    async def gather_all(tasks):
        return await tqdm.gather(*tasks)

    # Generate rollouts for each prompt
    l_responses = []
    for i in range(len(l_inputs)):
        # Generate n_samples rollouts for each input
        for _ in range(n_samples):
            l_responses.append(run_chat(l_inputs[i]["prompt"], include_logprobs=False))
    
    l_responses = asyncio.run(gather_all(l_responses))
    
    # Organize responses by input
    for i in range(len(l_inputs)):
        start_idx = i * n_samples
        end_idx = start_idx + n_samples
        l_inputs[i]["model_cot"] = l_responses[start_idx:end_idx]
    
    # Now get logprobs for the ground truth translations
    print("Computing logprobs for ground truth translations...")
    
    l_logprob_prompts = []
    l_assistant_token_starts = []
    
    for row in l_inputs:
        # Create prompt with ground truth as assistant response
        logprob_prompt = row["prompt"][:-1] if translation_prompt_type == "speaking_zero_shot" else row["prompt"]
        
        # First, get the tokens for the prompt WITHOUT the assistant response
        prompt_tokens = tokenizer.apply_chat_template(
            logprob_prompt,
            add_generation_prompt=True,  # Add the assistant prompt marker
            tokenize=True,
        )
        prompt_token_count = len(prompt_tokens)
        
        # Now add the assistant response
        logprob_prompt = logprob_prompt + [{
            "role": "assistant",
            "content": row["translated_solution"]
        }]
        # The assistant response starts after prompt_token_count tokens
        l_assistant_token_starts.append(prompt_token_count)
        l_logprob_prompts.append(logprob_prompt)
    
    # Get logprobs for ground truth
    async def get_logprobs_batch():
        tasks = []
        for prompt in l_logprob_prompts:
            # Use tokenizer to convert chat format to string
            prompt_str = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=False
            )
            
            async def get_completion_logprobs(prompt_text):
                for retry in range(20):
                    try:
                        async with rate_limit:
                            resp = await client.completions.create(
                                model=deployment_model_id,
                                prompt=prompt_text,
                                max_tokens=1,  # We just want logprobs, not generation
                                temperature=0,
                                logprobs=1,
                                echo=True  # Include prompt in response to get all logprobs
                            )
                            
                            if hasattr(resp.choices[0], 'logprobs'):
                                return resp.choices[0].logprobs
                            else:
                                return None
                                
                    except Exception as e:
                        print(f"Error getting logprobs on retry {retry}: {e}")
                        await asyncio.sleep(min(2 ** retry, 60))
                
                return None
            
            tasks.append(get_completion_logprobs(prompt_str))
        
        return await gather_all(tasks)
    
    logprobs_results = asyncio.run(get_logprobs_batch())
    
    # Process logprobs
    for i, logprobs_data in enumerate(logprobs_results):
        # Extract logprobs for the assistant response portion
        tokens = logprobs_data.tokens
        token_logprobs = logprobs_data.token_logprobs
        
        # Use the pre-calculated assistant token start position
        assistant_start_idx = l_assistant_token_starts[i]
        
        if assistant_start_idx < len(token_logprobs):
            gt_logprobs = token_logprobs[assistant_start_idx:]
            gt_tokens = tokens[assistant_start_idx:]
            
            # Filter out None values (first token doesn't have logprob)
            gt_logprobs = [lp if lp is not None else -100.0 for lp in gt_logprobs]
            
            l_inputs[i]["gt_logprobs"] = gt_logprobs
            l_inputs[i]["gt_logprob_tokens"] = gt_tokens
        else:
            # Fallback: token mismatch, try to extract what we can
            raise RuntimeError(f"Warning: Token count mismatch for input {i}. Expected start: {assistant_start_idx}, got {len(token_logprobs)} tokens total")

    df_output = pd.DataFrame(l_inputs)
    df_output.to_parquet(os.path.join(data_dir, "prompted_cot.parquet"))
    print(f"Saved results to {os.path.join(data_dir, 'prompted_cot.parquet')}")


def ensure_fireworks_deployment(config):
    pass


def tear_down_fireworks_deployment(config):
    pass