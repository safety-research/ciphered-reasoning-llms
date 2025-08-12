import pandas as pd
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple
import csv
import os
from datasets import load_dataset


def get_prm800k_cot():
    ds = load_dataset("Mai0313/prm800k")

    return [r['solution'] for r in ds['train']]
