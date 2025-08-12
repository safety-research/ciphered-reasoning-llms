import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from prompts.letter_substitutions import letter_to_word_with_dot_prompt


def get_translation_prompt(encoding_scheme_name):
    if encoding_scheme_name == "letter_to_word_with_dot":
        return letter_to_word_with_dot_prompt
    else:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme_name}")
