import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from encoding_schemes.letter_substitutions import letter_to_word_with_dot


def get_encoding_scheme(encoding_scheme_name):

    if encoding_scheme_name == "letter_to_word_with_dot":
        return letter_to_word_with_dot
    else:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme_name}")
