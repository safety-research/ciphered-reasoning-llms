import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from prompts.letter_substitutions import letter_to_word_with_dot_prompt, dot_between_chars
from prompts.identity import identity_prompt


def get_translation_prompt(encoding_scheme_name):
    if encoding_scheme_name == "letter_to_word_with_dot":
        return letter_to_word_with_dot_prompt
    elif encoding_scheme_name == "dot_between_chars":
        return dot_between_chars
    elif encoding_scheme_name == "identity_prompt":
        return identity_prompt
    else:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme_name}")
