import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from encoding_schemes.ciphers import (
    rot13_cipher,
    base64_cipher,
    base64_2x_cipher,
    base64_3x_cipher,
    caesar_cipher,
)
from encoding_schemes.compression import (
    gzip_to_bpe_encoded,
    gzip_to_base64_encoded,
)
from encoding_schemes.letter_permutations import (
    reverse_letters_in_each_word,
    random_permute_letters_in_each_word,
    swap_even_odd_letters_in_each_word,
    reverse_fibonacci_indices_in_each_word,
)
from encoding_schemes.destructive_mutations import (
    replace_80pct_letters_with_star,
    first_letter_of_each_word,
    first_token_of_each_word_model_tokenizer,
)
from encoding_schemes.letter_substitutions import (
    letter_to_word_with_dot,
    dot_between_chars,
    letter_to_poem_first_letter,
    inverse_letter_to_word_with_dot,
    inverse_dot_between_chars,
)


def get_encoding_scheme(encoding_scheme_name, config):
    # local callables for entries that depend on config / are identity
    identity = lambda x: x
    tokenizer = lambda s: first_token_of_each_word_model_tokenizer(
        s, config["experiment"]["experiment_params"]["model"]
    )

    encoding_map = {
        # substitutions / identity
        "identity": identity,
        "reverse_identity": identity,
        "speaking_identity": identity,
        "letter_to_word_with_dot": letter_to_word_with_dot,
        "reverse_letter_to_word_with_dot": letter_to_word_with_dot,
        "speaking_letter_to_word_with_dot": letter_to_word_with_dot,
        "dot_between_chars": dot_between_chars,
        "reverse_dot_between_chars": dot_between_chars,
        "speaking_dot_between_chars": dot_between_chars,
        "letter_to_poem_first_letter": letter_to_poem_first_letter,
        "reverse_letter_to_poem_first_letter": letter_to_poem_first_letter,
        # ciphers
        "rot13_cipher": rot13_cipher,
        "reverse_rot13_cipher": rot13_cipher,
        "base64_cipher": base64_cipher,
        "reverse_base64_cipher": base64_cipher,
        "base64_2x_cipher": base64_2x_cipher,
        "reverse_base64_2x_cipher": base64_2x_cipher,
        "base64_3x_cipher": base64_3x_cipher,
        "reverse_base64_3x_cipher": base64_3x_cipher,
        "caesar_cipher": caesar_cipher,
        "reverse_caesar_cipher": caesar_cipher,
        # compression
        "gzip_to_bpe_encoded": gzip_to_bpe_encoded,
        "reverse_gzip_to_bpe_encoded": gzip_to_bpe_encoded,
        "gzip_to_base64_encoded": gzip_to_base64_encoded,
        "reverse_gzip_to_base64_encoded": gzip_to_base64_encoded,
        # letter permutations
        "reverse_letters_in_each_word": reverse_letters_in_each_word,
        "reverse_reverse_letters_in_each_word": reverse_letters_in_each_word,
        "random_permute_letters_in_each_word": random_permute_letters_in_each_word,
        "reverse_random_permute_letters_in_each_word": random_permute_letters_in_each_word,
        "swap_even_odd_letters_in_each_word": swap_even_odd_letters_in_each_word,
        "reverse_swap_even_odd_letters_in_each_word": swap_even_odd_letters_in_each_word,
        "reverse_fibonacci_indices_in_each_word": reverse_fibonacci_indices_in_each_word,
        "reverse_reverse_fibonacci_indices_in_each_word": reverse_fibonacci_indices_in_each_word,
        # destructive mutations
        "replace_80pct_letters_with_star": replace_80pct_letters_with_star,
        "reverse_replace_80pct_letters_with_star": replace_80pct_letters_with_star,
        "first_letter_of_each_word": first_letter_of_each_word,
        "reverse_first_letter_of_each_word": first_letter_of_each_word,
        "first_token_of_each_word_model_tokenizer": tokenizer,
        "reverse_first_token_of_each_word_model_tokenizer": tokenizer,
    }

    try:
        return encoding_map[encoding_scheme_name]
    except KeyError:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme_name}")


def get_inverse_encoding_scheme(encoding_scheme_name, config):
    identity = lambda x: x

    encoding_map = {
        "speaking_identity": identity,
        "speaking_letter_to_word_with_dot": inverse_letter_to_word_with_dot,
        "speaking_dot_between_chars": inverse_dot_between_chars,
    }

    try:
        return encoding_map[encoding_scheme_name]
    except KeyError:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme_name}")


def is_async_encoding_scheme(encoding_scheme_name):
    return False
