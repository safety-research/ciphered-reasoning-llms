import pandas as pd
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple
import csv
import os
from datasets import load_dataset


def get_lmsys_chat_1m_1_turn_english():
    ds = load_dataset("lmsys/lmsys-chat-1m")

    ds = ds['train']
    ds = ds.filter(lambda r: r['language'] == 'English' and r['turn'] == 1, num_proc=16)

    return ds