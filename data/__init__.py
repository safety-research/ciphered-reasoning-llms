import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.designing_data_intensive_applications import get_designing_data_intensive_applications


def get_dataset(dataset_name):
    if dataset_name == "designing_data_intensive_applications":
        return get_designing_data_intensive_applications()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
