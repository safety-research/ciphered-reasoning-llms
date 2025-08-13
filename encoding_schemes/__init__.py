import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from encoding_schemes.ciphers import rot13_cipher, base64_cipher, base64_2x_cipher, base64_3x_cipher, caesar_cipher
from encoding_schemes.compression import gzip_to_bpe_encoded, gzip_to_base64_encoded
from encoding_schemes.letter_permutations import reverse_letters_in_each_word, random_permute_letters_in_each_word, swap_even_odd_letters_in_each_word, reverse_fibonacci_indices_in_each_word
from encoding_schemes.destructive_mutations import replace_80pct_letters_with_star, first_letter_of_each_word, first_token_of_each_word_model_tokenizer
from encoding_schemes.letter_substitutions import letter_to_word_with_dot, dot_between_chars, letter_to_poem_first_letter


def get_encoding_scheme(encoding_scheme_name, config):

    if encoding_scheme_name == "letter_to_word_with_dot":
        return letter_to_word_with_dot
    elif encoding_scheme_name == "identity_prompt":
        return lambda x: x
    elif encoding_scheme_name == "dot_between_chars":
        return dot_between_chars
    elif encoding_scheme_name == "rot13_cipher":
        return rot13_cipher
    elif encoding_scheme_name == "base64_cipher":
        return base64_cipher
    elif encoding_scheme_name == "base64_2x_cipher":
        return base64_2x_cipher
    elif encoding_scheme_name == "base64_3x_cipher":
        return base64_3x_cipher
    elif encoding_scheme_name == "caesar_cipher":
        return caesar_cipher
    elif encoding_scheme_name == "gzip_to_bpe_encoded":
        return gzip_to_bpe_encoded
    elif encoding_scheme_name == "gzip_to_base64_encoded":
        return gzip_to_base64_encoded
    elif encoding_scheme_name == "reverse_letters_in_each_word":
        return reverse_letters_in_each_word
    elif encoding_scheme_name == "random_permute_letters_in_each_word":
        return random_permute_letters_in_each_word
    elif encoding_scheme_name == "swap_even_odd_letters_in_each_word":
        return swap_even_odd_letters_in_each_word
    elif encoding_scheme_name == "reverse_fibonacci_indices_in_each_word":
        return reverse_fibonacci_indices_in_each_word
    elif encoding_scheme_name == "replace_80pct_letters_with_star":
        return replace_80pct_letters_with_star
    elif encoding_scheme_name == "first_letter_of_each_word":
        return first_letter_of_each_word
    elif encoding_scheme_name == "first_token_of_each_word_model_tokenizer":
        return lambda s: first_token_of_each_word_model_tokenizer(s, config["experiment"]["experiment_params"]["model"])
    elif encoding_scheme_name == "letter_to_poem_first_letter":
        return letter_to_poem_first_letter
    else:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme_name}")


def is_async_encoding_scheme(encoding_scheme_name):
    return False