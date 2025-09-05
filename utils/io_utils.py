import os
import sys
import ray
import shutil
import pandas as pd
import numpy as np
import string
import duckdb


@ray.remote(num_cpus=1)
def copy_file(config, from_template, to_template):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash

    experiment_hash = compute_experiment_hash(config)

    from_path = from_template.replace("__HASH__", experiment_hash)
    to_path = to_template.replace("__HASH__", experiment_hash)

    print(f"Copying {from_path} to {to_path}")

    shutil.copy(from_path, to_path)



def interleave_dataframes(l_dfs):
    return (
        pd.concat({ string.ascii_uppercase[i] : l_dfs[i] for i in range(len(l_dfs)) })
        .swaplevel(0, 1)                   # make (row_idx, source)
        .sort_index(level=[0, 1])          # A1, B1, C1, A2, B2, C2...
        .reset_index(level=1, drop=True)   # drop the source level
        .reset_index(drop=True)
    )



@ray.remote(num_cpus=1, memory=32 * 1024 * 1024 * 1024)
def combine_parquet_files(config, to_template, shuffle=False, index_lockstep=False, **kwargs):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from utils.io_utils import interleave_dataframes, read_large_parquet

    experiment_hash = compute_experiment_hash(config)

    l_files = []

    for _, file_name in kwargs.items():
        file_name = file_name.replace("__HASH__", experiment_hash)

        l_files.append(read_large_parquet(file_name))
        print(f"Read {file_name}")

    df_combined = pd.concat(l_files, ignore_index=True)
    if shuffle:
        df_combined = df_combined.sample(frac=1.0, random_state=42)
    elif index_lockstep:
        n_rows = [len(df) for df in l_files]
        n_rows = set(n_rows)

        assert len(n_rows) == 1

        n_rows = next(iter(n_rows))

        indices = list(range(n_rows))

        np.random.seed(42)
        np.random.shuffle(indices)

        l_files = [df.iloc[indices] for df in l_files]
        
        df_combined = interleave_dataframes(l_files)
    else:
        df_combined = pd.concat(l_files, ignore_index=True)

    to_path = to_template.replace("__HASH__", experiment_hash)

    os.makedirs(os.path.dirname(to_path), exist_ok=True)
    df_combined.to_parquet(to_path)

    print(f"Wrote {len(df_combined)} rows {to_path}")


@ray.remote(num_cpus=1, memory=64 * 1024 * 1024 * 1024)
def write_token_count(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from utils.io_utils import read_large_parquet

    from transformers import AutoTokenizer

    model = config["experiment"]["experiment_params"]["model"]
    if "gpt" in model or "claude" in model:
        print(f"Overriding tokenizer for {model} with gpt-oss 120b tokenizer because it was detected as a GPT/Claude model!")
        model = "openai/gpt-oss-120b"

    tokenizer = AutoTokenizer.from_pretrained(model)

    experiment_hash = compute_experiment_hash(config)

    for suffix in ['', '_train']:
        path = os.path.join('output', experiment_hash, 'data', f'sft{suffix}.parquet')

        df = read_large_parquet(path)
        df['num_tokens'] = df['messages'].map(lambda x: len(tokenizer.apply_chat_template(x)) )

        df.to_parquet(path)


def read_large_parquet(path):
    return duckdb.query(f"SELECT * FROM read_parquet('{path}')").to_df()


@ray.remote(num_cpus=1, memory=64 * 1024 * 1024 * 1024)
def take_first_n_rows(config, df_path, n_rows):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash
    from utils.io_utils import read_large_parquet

    experiment_hash = compute_experiment_hash(config)

    df_path = df_path.replace("__HASH__", experiment_hash)

    df = read_large_parquet(df_path)

    df = df.iloc[:n_rows]

    df.to_parquet(df_path)


@ray.remote(num_cpus=1, memory=64 * 1024 * 1024 * 1024)
def combine_translation_math_cot_dfs(config):
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from orchestration.experiment_meta_saver import compute_experiment_hash

    experiment_hash = compute_experiment_hash(config)

    df_gt_translation = pd.read_parquet(os.path.join("output", experiment_hash, "data", "ground_truth_translation.parquet"))
    df_prompted_cot = pd.read_parquet(os.path.join("output", experiment_hash, "data", "prompted_cot.parquet"))
    df_math_scores = pd.read_parquet(os.path.join("output", experiment_hash, "data", "math_scores.parquet"))
    df_prompted_translation = pd.read_parquet(os.path.join("output", experiment_hash, "data", "prompted_translation.parquet"))
    df_bleu_scores = pd.read_parquet(os.path.join("output", experiment_hash, "data", "bleu_scores.parquet"))

    l_dfs = [df_gt_translation, df_prompted_cot, df_math_scores, df_prompted_translation, df_bleu_scores]
    for i in range(len(l_dfs)):
        for j in range(i + 1, len(l_dfs)):
            if len(l_dfs[i]) != len(l_dfs[j]):
                raise Exception(f"{i} and {j} had mismatched len {len(l_dfs[i])} and {len(l_dfs[j])}")

    # df_gt_translation kept as is
    d_prompted_cot_cols = {
        "prompt": "cot_prompt",
        "gt_logprobs": "cot_gt_logprobs",
        "gt_logprob_tokens": "cot_gt_logprob_tokens",
        "model_cot": "generated_cots",
        "followed_encoding_style": "generated_cot_adhered_encoding_style"
    }
    df_prompted_cot = df_prompted_cot[list(d_prompted_cot_cols.keys())]
    df_prompted_cot = df_prompted_cot.rename(columns=d_prompted_cot_cols)

    d_math_score_cols = {
        "is_corrects": "generated_cot_is_correct"
    }
    df_math_scores = df_math_scores[list(d_math_score_cols.keys())]
    df_math_scores = df_math_scores.rename(columns=d_math_score_cols)

    d_prompted_translation_cols = {
        # Encoded form (reverse translation)
        "reference_text": "source_backtranslation_text",
        "gt_translation": "target_backtranslation_text",
        "prompt": "backtranslation_prompt",
        "model_translations": "generated_backtranslations",
        "gt_logprobs": "backtranslation_gt_logprobs",
        "gt_logprob_tokens": "backtranslation_gt_logprob_tokens"
    }
    df_prompted_translation = df_prompted_translation[list(d_prompted_translation_cols.keys())]
    df_prompted_translation = df_prompted_translation.rename(columns=d_prompted_translation_cols)

    d_bleu_score_cols = {
        "bleu_scores": "backtranslation_bleu_scores"
    }
    df_bleu_scores = df_bleu_scores[list(d_bleu_score_cols.keys())]
    df_bleu_scores = df_bleu_scores.rename(columns=d_bleu_score_cols)

    l_dfs_to_concat = [df_gt_translation, df_prompted_cot, df_math_scores, df_prompted_translation, df_bleu_scores]
    assert len(l_dfs_to_concat) == len(l_dfs)
    df_final = pd.concat(l_dfs_to_concat, axis='columns')

    target_path = os.path.join("output", experiment_hash, "data", "joined_output.parquet")
    df_final.to_parquet(target_path)
    print(f"Wrote {len(df_final)} rows to {target_path}")

