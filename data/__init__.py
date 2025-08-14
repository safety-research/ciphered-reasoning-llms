import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.designing_data_intensive_applications import get_designing_data_intensive_applications
from data.prm800k import get_prm800k_cot, get_prm800k_raw


def get_dataset(dataset_name):
    if dataset_name == "designing_data_intensive_applications":
        return get_designing_data_intensive_applications()
    elif dataset_name == "prm800k_cot":
        return get_prm800k_cot()
    elif dataset_name == "prm800k_raw":
        return get_prm800k_raw()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
