import pandas as pd
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple
import csv
import os
from datasets import load_dataset


def get_numina_math_cot_raw():
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from verl.utils.reward_score.math import (
        compute_score,
        last_boxed_only_string,
        remove_boxed,
    )

    ds = load_dataset("AI-MO/NuminaMath-CoT")

    def column_remapper(r):
        return {
            "problem": r["problem"],
            "solution": r["solution"],
            "answer": last_boxed_only_string(r["solution"]),
        }

    def box_remover(r):
        try:
            r["answer"] = remove_boxed(r["answer"])
        except AssertionError as e:
            print(e)
            return r
        return r

    return (
        ds["train"]
        .map(column_remapper, num_proc=16)
        .filter(lambda r: r["answer"] is not None)
        .map(box_remover, num_proc=16)
        .shuffle(seed=42)
    )
