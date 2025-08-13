import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Prompts for substitutions / identity
from prompts.identity import identity_prompt
from prompts.letter_substitutions import (
    letter_to_word_with_dot_prompt,
    dot_between_chars,  # kept as-is to match your existing import/name
    letter_to_poem_first_letter_prompt,
)

# Prompts for classic ciphers
from prompts.ciphers import (
    rot13_cipher_prompt,
    base64_cipher_prompt,
    base64_2x_cipher_prompt,
    base64_3x_cipher_prompt,
    caesar_cipher_prompt,
)

# Prompts for compression-based transforms
from prompts.compression import (
    gzip_to_bpe_encoded_prompt,
    gzip_to_base64_encoded_prompt,
)

# Prompts for letter permutations
from prompts.letter_permutations import (
    reverse_letters_in_each_word_prompt,
    random_permute_letters_in_each_word_prompt,
    swap_even_odd_letters_in_each_word_prompt,
    reverse_fibonacci_indices_in_each_word_prompt,
)

# Prompts for destructive mutations
from prompts.destructive_mutations import (
    replace_80pct_letters_with_star_prompt,
    first_letter_of_each_word_prompt,
    first_token_of_each_word_model_tokenizer_prompt,
)


def get_translation_prompt(encoding_scheme_name):
    if encoding_scheme_name == "letter_to_word_with_dot":
        return letter_to_word_with_dot_prompt
    elif encoding_scheme_name == "identity_prompt":
        return identity_prompt
    elif encoding_scheme_name == "dot_between_chars":
        return dot_between_chars

    # ciphers
    elif encoding_scheme_name == "rot13_cipher":
        return rot13_cipher_prompt
    elif encoding_scheme_name == "base64_cipher":
        return base64_cipher_prompt
    elif encoding_scheme_name == "base64_2x_cipher":
        return base64_2x_cipher_prompt
    elif encoding_scheme_name == "base64_3x_cipher":
        return base64_3x_cipher_prompt
    elif encoding_scheme_name == "caesar_cipher":
        return caesar_cipher_prompt

    # compression
    elif encoding_scheme_name == "gzip_to_bpe_encoded":
        return gzip_to_bpe_encoded_prompt
    elif encoding_scheme_name == "gzip_to_base64_encoded":
        return gzip_to_base64_encoded_prompt

    # letter permutations
    elif encoding_scheme_name == "reverse_letters_in_each_word":
        return reverse_letters_in_each_word_prompt
    elif encoding_scheme_name == "random_permute_letters_in_each_word":
        return random_permute_letters_in_each_word_prompt
    elif encoding_scheme_name == "swap_even_odd_letters_in_each_word":
        return swap_even_odd_letters_in_each_word_prompt
    elif encoding_scheme_name == "reverse_fibonacci_indices_in_each_word":
        return reverse_fibonacci_indices_in_each_word_prompt

    # destructive mutations
    elif encoding_scheme_name == "replace_80pct_letters_with_star":
        return replace_80pct_letters_with_star_prompt
    elif encoding_scheme_name == "first_letter_of_each_word":
        return first_letter_of_each_word_prompt
    elif encoding_scheme_name == "first_token_of_each_word_model_tokenizer":
        return first_token_of_each_word_model_tokenizer_prompt

    # substitutions
    elif encoding_scheme_name == "letter_to_poem_first_letter":
        return letter_to_poem_first_letter_prompt

    else:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme_name}")
