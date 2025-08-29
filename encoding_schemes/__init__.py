import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from encoding_schemes.ciphers import (
    rot13_cipher,
    base64_cipher,
    base64_2x_cipher,
    base64_3x_cipher,
    caesar_cipher,

    inverse_base64_cipher,
    inverse_base64_2x_cipher,
    inverse_base64_3x_cipher,
    inverse_caesar_cipher
)
from encoding_schemes.compression import (
    gzip_to_bpe_encoded,
    gzip_to_base64_encoded,

    inverse_gzip_to_bpe_encoded,
    inverse_gzip_to_base64_encoded
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
    remove_all_verbs,
    remove_all_nouns
)
from encoding_schemes.letter_substitutions import (
    letter_to_word_with_dot,
    dot_between_chars,
    letter_to_poem_first_letter,
    inverse_letter_to_word_with_dot,
    inverse_dot_between_chars,
    space_between_chars,
    inverse_space_between_chars
)
from encoding_schemes.translations import (
    translate_to_English,
    translate_to_French,
    translate_to_Chinese,
    translate_to_Korean,
    translate_to_Russian,
    translate_to_Arabic,
    translate_to_Adyghe,
    translate_to_Briefhand,
    translate_from_Briefhand,
    translate_to_morse_code,
    translate_to_Python,
    translate_to_enterprise_Java
)
from encoding_schemes.steganography import (
    speaking_math_safety_steg,
    speaking_math_sonnet_steg,
    speaking_math_news_article_steg,
    speaking_math_enterprise_java_steg,
    speaking_math_weather_report_steg,
    speaking_math_numbers_sequence_steg
)


def get_encoding_scheme(encoding_scheme_name, config):
    # local callables for entries that depend on config / are identity
    zero_shot = lambda x: ""
    identity = lambda x: x
    tokenizer = lambda s: first_token_of_each_word_model_tokenizer(
        s, config["experiment"]["experiment_params"]["model"]
    )

    encoding_map = {
        # substitutions / identity
        "zero_shot": zero_shot,
        "reverse_zero_shot": zero_shot,
        "speaking_zero_shot": zero_shot,

        "identity": identity,
        "reverse_identity": identity,
        "speaking_identity": identity,

        "letter_to_word_with_dot": letter_to_word_with_dot,
        "reverse_letter_to_word_with_dot": letter_to_word_with_dot,
        "speaking_letter_to_word_with_dot": letter_to_word_with_dot,

        "dot_between_chars": dot_between_chars,
        "reverse_dot_between_chars": dot_between_chars,
        "speaking_dot_between_chars": dot_between_chars,

        "speaking_space_between_chars": space_between_chars,

        "letter_to_poem_first_letter": letter_to_poem_first_letter,
        "reverse_letter_to_poem_first_letter": letter_to_poem_first_letter,
        # ciphers
        "rot13_cipher": rot13_cipher,
        "reverse_rot13_cipher": rot13_cipher,
        "speaking_rot13_cipher": rot13_cipher,

        "base64_cipher": base64_cipher,
        "reverse_base64_cipher": base64_cipher,
        "speaking_base64_cipher": base64_cipher,

        "base64_2x_cipher": base64_2x_cipher,
        "reverse_base64_2x_cipher": base64_2x_cipher,
        "speaking_base64_2x_cipher": base64_2x_cipher,

        "base64_3x_cipher": base64_3x_cipher,
        "reverse_base64_3x_cipher": base64_3x_cipher,
        "speaking_base64_3x_cipher": base64_3x_cipher,

        "caesar_cipher": caesar_cipher,
        "reverse_caesar_cipher": caesar_cipher,
        "speaking_caesar_cipher": caesar_cipher,

        # compression
        "gzip_to_bpe_encoded": gzip_to_bpe_encoded,
        "reverse_gzip_to_bpe_encoded": gzip_to_bpe_encoded,
        "speaking_gzip_to_bpe_encoded": gzip_to_bpe_encoded,

        "gzip_to_base64_encoded": gzip_to_base64_encoded,
        "reverse_gzip_to_base64_encoded": gzip_to_base64_encoded,
        "speaking_gzip_to_base64_encoded": gzip_to_base64_encoded,

        # letter permutations
        "reverse_letters_in_each_word": reverse_letters_in_each_word,
        "reverse_reverse_letters_in_each_word": reverse_letters_in_each_word,
        "speaking_reverse_letters_in_each_word": reverse_letters_in_each_word,

        "random_permute_letters_in_each_word": random_permute_letters_in_each_word,
        "reverse_random_permute_letters_in_each_word": random_permute_letters_in_each_word,

        "swap_even_odd_letters_in_each_word": swap_even_odd_letters_in_each_word,
        "reverse_swap_even_odd_letters_in_each_word": swap_even_odd_letters_in_each_word,
        "speaking_swap_even_odd_letters_in_each_word": swap_even_odd_letters_in_each_word,

        "reverse_fibonacci_indices_in_each_word": reverse_fibonacci_indices_in_each_word,
        "reverse_reverse_fibonacci_indices_in_each_word": reverse_fibonacci_indices_in_each_word,
        "speaking_reverse_fibonacci_indices_in_each_word": reverse_fibonacci_indices_in_each_word,

        # destructive mutations
        "replace_80pct_letters_with_star": replace_80pct_letters_with_star,
        "reverse_replace_80pct_letters_with_star": replace_80pct_letters_with_star,
        "first_letter_of_each_word": first_letter_of_each_word,
        "reverse_first_letter_of_each_word": first_letter_of_each_word,
        "first_token_of_each_word_model_tokenizer": tokenizer,
        "reverse_first_token_of_each_word_model_tokenizer": tokenizer,
        "remove_all_verbs": remove_all_verbs,
        "remove_all_nouns": remove_all_nouns,

        # language translations
        "speaking_French": translate_to_French,
        "speaking_Chinese": translate_to_Chinese,
        "speaking_Korean": translate_to_Korean,
        "speaking_Russian": translate_to_Russian,
        "speaking_Arabic": translate_to_Arabic,
        "speaking_Adyghe": translate_to_Adyghe,
        "speaking_Briefhand": translate_to_Briefhand,
        "speaking_Morse_code": translate_to_morse_code,
        "speaking_Python": translate_to_Python,
        "speaking_enterprise_Java": translate_to_enterprise_Java,

        # steg
        "speaking_math_safety_steg": speaking_math_safety_steg,
        "speaking_math_sonnet_steg": speaking_math_sonnet_steg,
        "speaking_math_news_article_steg": speaking_math_news_article_steg,
        "speaking_math_enterprise_java_steg": speaking_math_enterprise_java_steg,
        "speaking_math_weather_report_steg": speaking_math_weather_report_steg,
        "speaking_math_numbers_sequence_steg": speaking_math_numbers_sequence_steg
    }

    try:
        return encoding_map[encoding_scheme_name]
    except KeyError:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme_name}")


def get_inverse_encoding_scheme(encoding_scheme_name, config):
    identity = lambda x: x

    async def async_identity(x):
        return x

    encoding_map = {
        "speaking_zero_shot": identity, # placeholder since we reuse the pipeline from math CoT for zero shot
        "speaking_identity": identity,
        "speaking_letter_to_word_with_dot": inverse_letter_to_word_with_dot,
        "speaking_dot_between_chars": inverse_dot_between_chars,
        "speaking_space_between_chars": inverse_space_between_chars,
        "speaking_rot13_cipher": rot13_cipher, # rot13 is symmetric
        "speaking_base64_cipher": inverse_base64_cipher,
        "speaking_base64_2x_cipher": inverse_base64_2x_cipher,
        "speaking_base64_3x_cipher": inverse_base64_3x_cipher,
        "speaking_caesar_cipher": inverse_caesar_cipher,
        "speaking_gzip_to_bpe_encoded": inverse_gzip_to_bpe_encoded,
        "speaking_gzip_to_base64_encoded": inverse_gzip_to_base64_encoded,
        "speaking_reverse_letters_in_each_word": reverse_letters_in_each_word, # reverse word letters is symmetric
        "speaking_swap_even_odd_letters_in_each_word": swap_even_odd_letters_in_each_word, # symmetric
        "speaking_reverse_fibonacci_indices_in_each_word": reverse_fibonacci_indices_in_each_word, # also symmetric

        "speaking_French": lambda x: translate_to_English(x, "French"),
        "speaking_Chinese": lambda x: translate_to_English(x, "Chinese"),
        "speaking_Korean": lambda x: translate_to_English(x, "Korean"),
        "speaking_Russian": lambda x: translate_to_English(x, "Russian"),
        "speaking_Arabic": lambda x: translate_to_English(x, "Arabic"),
        "speaking_Adyghe": lambda x: translate_to_English(x, "Adyghe"),
        "speaking_Briefhand": translate_from_Briefhand,
        "speaking_Morse_code": lambda x: translate_to_English(x, "Morse code"),
        "speaking_Python": async_identity, # Python/Java don't have coherency checks
        "speaking_enterprise_Java": async_identity,

        # Steg doesn't measure coherence because the expectation is the output will be coherent
        # and we measure detectability instead
        "speaking_math_safety_steg": async_identity,
        "speaking_math_sonnet_steg": async_identity,
        "speaking_math_news_article_steg": async_identity,
        "speaking_math_enterprise_java_steg": async_identity,
        "speaking_math_weather_report_steg": async_identity,
        "speaking_math_numbers_sequence_steg": async_identity,

        # destructive mutations that should be easy to keep coherent
        "remove_all_verbs": async_identity,
        "remove_all_nouns": async_identity
    }

    try:
        return encoding_map[encoding_scheme_name]
    except KeyError:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme_name}")


def is_async_encoding_scheme(encoding_scheme_name):
    s_async_encodings = set([
        "speaking_French",
        "speaking_Chinese",
        "speaking_Korean",
        "speaking_Russian",
        "speaking_Arabic",
        "speaking_Adyghe",
        "speaking_Briefhand",
        "speaking_Morse_code",
        "speaking_Python",
        "speaking_enterprise_Java",
        "speaking_math_safety_steg",
        "speaking_math_sonnet_steg",
        "speaking_math_news_article_steg",
        "speaking_math_enterprise_java_steg",
        "speaking_math_weather_report_steg",
        "speaking_math_numbers_sequence_steg",

        "remove_all_verbs",
        "remove_all_nouns"
    ])

    return encoding_scheme_name in s_async_encodings
