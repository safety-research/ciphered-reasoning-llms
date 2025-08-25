import copy
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
        try:
            total_tokens += len(encoding.encode(msg.get("role", ""), disallowed_special=()))
            total_tokens += len(encoding.encode(msg.get("content", ""), disallowed_special=()))
        except Exception as e:
            print(e)
            total_tokens += 1000
        except ValueError as e:
            print(e)
            total_tokens += 1000
    return total_tokens


@ray.remote(num_cpus=1)
def generate_ground_truth_translation(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from encoding_schemes import get_encoding_scheme, is_async_encoding_scheme
    from data import get_dataset
    from orchestration.experiment_meta_saver import compute_experiment_hash

    fn_encoding_scheme = get_encoding_scheme(config["experiment"]["experiment_params"]["encoding_scheme"], config)

    dataset = get_dataset(config["experiment"]["experiment_params"]["dataset"])

    experiment_hash = compute_experiment_hash(config)
    target_path = os.path.join("output", experiment_hash, "data", "ground_truth_translation.parquet")

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    async def gather_all(tasks):
        return await asyncio.gather(*tasks)

    async def translate_conversation(conversation):
        conversation = copy.deepcopy(conversation)
        conversation = list(conversation)

        for msg in conversation:
            if msg['role'] == 'assistant':
                if is_async_encoding_scheme(config["experiment"]["experiment_params"]["encoding_scheme"]):
                    msg['content'] = await fn_encoding_scheme(msg['content'])
                else:
                    msg['content'] = fn_encoding_scheme(msg['content'])

        return conversation

    translated_conversation = [translate_conversation(r["conversation"]) for r in dataset]
    translated_conversation = asyncio.run(gather_all(translated_conversation))

    df = pd.DataFrame(
        {
            "reference_conversation": [r['conversation'] for r in dataset],
            "translated_conversation": translated_conversation
        }
    )

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

    def format_conversation(conversation):
        s = ""

        for msg in conversation:
            s += f"{msg['role']}: {msg['content']}\n"

        return s


    for i, row in df.iterrows():
        df_sample = df_sample_group[df_sample_group["translated_conversation"] != row["translated_conversation"]]
        df_sample = df_sample.sample(n=n_few_shot_examples, random_state=42)

        s = "\n"
        for j, sample_row in df_sample.iterrows():
            s += (
                f"Example {j + 1}. {format_conversation(sample_row['translated_conversation'])}"
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
        target_path = os.path.join("output", experiment_hash, "data", f"ground_truth_translation{suffix}.parquet")
        df = pd.read_parquet(target_path)

        df["len"] = df["translated_conversation"].map(lambda x: len(x[-1]['content']))
        df_sample_group = df.sort_values("len").head(100)
        df = df.drop(columns=["len"])

        df["few_shot_examples"] = get_few_shot_examples(df, df_sample_group, config)
        df.to_parquet(target_path)


@ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 32)
def generate_sft_dataset(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts import get_translation_prompt
    from speaking.conversational_chat_runner import count_tokens_from_messages

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

            row_translation_prompt = [{
                "role": "system",
                "content": row_translation_prompt
            }]

            l_inputs.append(
                {
                    "reference_conversation": row['reference_conversation'],
                    "messages": row_translation_prompt + list(row['translated_conversation'])
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

    def build_input(conversation):
        ret = []
        for msg in conversation:
            if msg['role'] == 'assistant':
                return ret
            else:
                ret.append(msg)
        return ret

    l_inputs = []
    for i, row in ground_truth_translation.iterrows():

        row_translation_prompt = translation_prompt
        if config["experiment"]["experiment_params"].get("n_few_shot_examples", 0):
            row_translation_prompt += "\n" + row["few_shot_examples"]
        row_translation_prompt = [{
            "role": "system",
            "content": row_translation_prompt
        }]

        l_inputs.append(
            {
                "reference_conversation": row['reference_conversation'],
                "translated_conversation": row_translation_prompt + list(row['translated_conversation']),
                "prompt": row_translation_prompt + build_input(row['translated_conversation'])
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
    sampling_params = SamplingParams(
        temperature=config["experiment"]["experiment_params"]["sampling_params"]["temperature"],
        max_tokens=4096,
        n=config["experiment"]["experiment_params"]["sampling_params"]["n"],
    )

    outputs = llm.chat([r["prompt"] for r in l_inputs], sampling_params=sampling_params, use_tqdm=True)

    l_input_token_lens = [len(o.prompt_token_ids) for o in outputs]
    for i, output in enumerate(outputs):
        l_inputs[i]["response"] = [choice.text for choice in output.outputs]

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
            row['translated_conversation']
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
    df_output.to_parquet(os.path.join("output", experiment_hash, "data", "prompted_chat.parquet"))

    kill_vllm_process(llm)


@ray.remote(num_cpus=1, num_gpus=2, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def judge_cot_style_adherence(config):
    from vllm import LLM, SamplingParams

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts.translation.judge import followed_encoding_style_judge
    from prompts import get_translation_prompt
    from utils.vllm import kill_vllm_process

    experiment_hash = compute_experiment_hash(config)

    generated_cot_path = os.path.join("output", experiment_hash, "data", "prompted_chat.parquet")
    df_generated_responses = pd.read_parquet(generated_cot_path)

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
    for (_, generated_cot_row), (_, sft_row) in zip(df_generated_responses.iterrows(), df_sft.iterrows()):
        sft_reference = sft_row['messages'][-1]['content']

        for response in generated_cot_row['response']:
            l_judge_prompts.append([{"role": "system", "content": "/no_think"}, {"role": "user", "content": followed_encoding_style_judge + f"\n<text>{cot}</text>\n<reference_text>{sft_reference}</reference_text>"}])

    judge_sampling_params = SamplingParams(max_tokens=1024)
    outputs = llm.chat(l_judge_prompts, sampling_params=judge_sampling_params, use_tqdm=True)
    outputs_idx = 0
    l_judge_scores = []

    for responses in df_generated_responses['response']:
        l_instance_scores = []
        for response in responses:
            text = outputs[outputs_idx].outputs[0].text
            outputs_idx += 1

            search_result = re.search("<answer>(.*?)</answer>", text)
            if search_result:
                l_instance_scores.append(1.0 if search_result.group(1) == "Yes" else 0.0)
            else:
                l_instance_scores.append(0.0)

        l_judge_scores.append(l_instance_scores)

    df_generated_responses["followed_encoding_style"] = l_judge_scores

    df_generated_responses.to_parquet(generated_cot_path)

    kill_vllm_process(llm)


@ray.remote(num_cpus=1, num_gpus=2, retry_exceptions=True, memory=1024 * 1024 * 1024 * 32)
def judge_cot_encoding_English_coherence(config):
    from vllm import LLM, SamplingParams

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from prompts.translation.judge import coherent_english_judge
    from encoding_schemes import get_inverse_encoding_scheme
    from utils.vllm import kill_vllm_process

    experiment_hash = compute_experiment_hash(config)

    target_path = os.path.join("output", experiment_hash, "data", "prompted_chat.parquet")

    df = pd.read_parquet(target_path)

    fn_inverse_encoding_scheme = get_inverse_encoding_scheme(
        config["experiment"]["experiment_params"]["encoding_scheme"], config
    )

    l_inverted_response = [[fn_inverse_encoding_scheme(response) for response in responses] for responses in df["response"]]

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
    for responses in l_inverted_response:
        for response in responses:
            l_judge_prompts.append([{"role": "system", "content": "/no_think"}, {"role": "user", "content": coherent_english_judge + f"\n<text>{cot}</text>"}])

    judge_sampling_params = SamplingParams(max_tokens=1024)
    outputs = llm.chat(l_judge_prompts, sampling_params=judge_sampling_params, use_tqdm=True)
    outputs_idx = 0
    l_judge_scores = []

    for cots in l_inverted_response:
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
    df["decoded_response"] = l_inverted_response

    df.to_parquet(target_path)

    kill_vllm_process(llm)
