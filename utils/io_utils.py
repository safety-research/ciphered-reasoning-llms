import os
import sys
import ray
import shutil
import pandas as pd
import numpy as np
import string


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
    from utils.io_utils import interleave_dataframes

    experiment_hash = compute_experiment_hash(config)

    l_files = []

    for _, file_name in kwargs.items():
        file_name = file_name.replace("__HASH__", experiment_hash)

        l_files.append(pd.read_parquet(file_name))
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
