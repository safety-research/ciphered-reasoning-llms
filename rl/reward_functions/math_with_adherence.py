import os
import sys
import numpy as np
import re
from verl.utils.reward_score.math import compute_score, last_boxed_only_string, remove_boxed

def extract_model_answer(model_response):
    try:
        if "\\boxed" in model_response:
            response = last_boxed_only_string(model_response)
            if response:
                response = remove_boxed(response)
                return response
            else:
                return ""
    except Exception as e:
        print(e)
        return ""


def is_math_answer_correct(parsed_answer, ref_answer):
    try:
        if "=" in parsed_answer:
            parsed_answer = parsed_answer.split("=")[1].strip()

        if "boxed" not in parsed_answer:
            parsed_answer = f"\\boxed{{{parsed_answer}}}"

        return compute_score(parsed_answer, ref_answer)
    except Exception as e:
        print(e)
        return 0.0



d_letter_to_word_with_dot_mapping = {
    "A": "Apple",
    "B": "Busy",
    "C": "Candle",
    "D": "Dragon",
    "E": "Echo",
    "F": "Forest",
    "G": "Galaxy",
    "H": "Harbor",
    "I": "Island",
    "J": "Journey",
    "K": "Kite",
    "L": "Lantern",
    "M": "Meadow",
    "N": "North",
    "O": "Ocean",
    "P": "Prism",
    "Q": "Quartz",
    "R": "River",
    "S": "Sunset",
    "T": "Thunder",
    "U": "Universe",
    "V": "Velvet",
    "W": "Willow",
    "X": "Xylophone",
    "Y": "Yellow",
    "Z": "Zephyr",
}



def calculate_letter_to_word_with_dot_adherence(s):
    s = s.replace("boxed", "")

    orig_len = len(s)

    for word in d_letter_to_word_with_dot_mapping.values():
        s = s.replace(word, "")

    s = s.replace(".", "")
    s = s.replace(" ", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace("\\", "")
    s = s.replace("[", "")
    s = s.replace("]", "")

    return len(s) / orig_len < 0.05


def math_adherent_and_correct_reward_letter_to_word_with_dot(data_sources, solution_strs, ground_truths, extra_infos):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from rl.reward_functions.math_with_adherence import extract_model_answer, is_math_answer_correct, calculate_letter_to_word_with_dot_adherence

    n = len(solution_strs)

    l_rewards = []

    for i in range(n):
        correct_score = is_math_answer_correct(extract_model_answer(solution_strs[i]), ground_truths[i])
        formatting_score = 1.0 if "boxed" in solution_strs[i] else 0.0
        adherence_score = 1.0 if calculate_letter_to_word_with_dot_adherence(solution_strs[i]) else -1.0

        l_rewards.append({
            'score': adherence_score * correct_score + (formatting_score * 0.1),
            'adherence_score': adherence_score,
            'correct_score': correct_score,
            'formatting_score': formatting_score
        })

    return l_rewards


def calculate_dot_between_chars_adherence(s):
    s = s.replace("boxed", "")

    num_dots = len([c for c in s if c == '.'])

    return num_dots / len(s) >= 0.45


def math_adherent_and_correct_reward_dot_between_chars(data_sources, solution_strs, ground_truths, extra_infos):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from rl.reward_functions.math_with_adherence import extract_model_answer, is_math_answer_correct, calculate_dot_between_chars_adherence

    n = len(solution_strs)

    l_rewards = []

    for i in range(n):
        correct_score = is_math_answer_correct(extract_model_answer(solution_strs[i]), ground_truths[i])
        formatting_score = 1.0 if "boxed" in solution_strs[i] else 0.0
        adherence_score = 1.0 if calculate_dot_between_chars_adherence(solution_strs[i]) else -1.0

        l_rewards.append({
            'score': adherence_score * correct_score + (formatting_score * 0.1),
            'adherence_score': adherence_score,
            'correct_score': correct_score,
            'formatting_score': formatting_score
        })

    return l_rewards


def math_adherent_and_correct_reward_base64(data_sources, solution_strs, ground_truths, extra_infos):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from rl.reward_functions.math_with_adherence import extract_model_answer, is_math_answer_correct
    from encoding_schemes.ciphers import base64_valid_percentage

    n = len(solution_strs)

    l_rewards = []

    for i in range(n):
        correct_score = is_math_answer_correct(extract_model_answer(solution_strs[i]), ground_truths[i])
        formatting_score = 1.0 if "boxed" in solution_strs[i] else 0.0
        adherence_score = 1.0 if base64_valid_percentage(solution_strs[i]) >= 0.95 else -1.0

        l_rewards.append({
            'score': adherence_score * correct_score + (formatting_score * 0.1),
            'adherence_score': adherence_score,
            'correct_score': correct_score,
            'formatting_score': formatting_score
        })

    return l_rewards


def math_correct_reward(data_sources, solution_strs, ground_truths, extra_infos):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from rl.reward_functions.math_with_adherence import extract_model_answer, is_math_answer_correct

    n = len(solution_strs)

    l_rewards = []

    for i in range(n):
        correct_score = is_math_answer_correct(extract_model_answer(solution_strs[i]), ground_truths[i])
        formatting_score = 1.0 if "boxed" in solution_strs[i] else 0.0
        adherence_score = 1.0

        l_rewards.append({
            'score': adherence_score * correct_score + (formatting_score * 0.1),
            'adherence_score': adherence_score,
            'correct_score': correct_score,
            'formatting_score': formatting_score
        })

    return l_rewards
