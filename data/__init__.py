import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.designing_data_intensive_applications import get_designing_data_intensive_applications
from data.prm800k import get_prm800k_cot, get_prm800k_raw
from data.lmsys_chat_1m import get_lmsys_chat_1m_1_turn_english


def get_dataset(dataset_name):
    if dataset_name == "designing_data_intensive_applications":
        return get_designing_data_intensive_applications()
    elif dataset_name == "prm800k_cot":
        return get_prm800k_cot()
    elif dataset_name == "prm800k_raw":
        return get_prm800k_raw()
    elif dataset_name == "lmsys_chat_1m_1_turn_english":
        return get_lmsys_chat_1m_1_turn_english()
    elif dataset_name == "lmsys_chat_1m_1_turn_english_subset":
        return get_lmsys_chat_1m_1_turn_english().select(range(100000))
    elif dataset_name == "lmsys_chat_1m_1_turn_english_debug":
        return get_lmsys_chat_1m_1_turn_english().select(range(1000))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
