import pandas as pd
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple
import csv
import os
from datasets import load_dataset


def get_zebra_bench_raw():
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'raw', 'zebra_bench/zebra_bench_transcripts.parquet'))
    df = df.rename(columns={
        'reasoning': 'solution',
        'solution': 'answer'
    })

    return df.to_dict('records')