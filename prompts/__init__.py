import os
import sys
from difflib import get_close_matches

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Prompts for substitutions / identity
from prompts.translation.identity import (
    identity_prompt,
    zero_shot_prompt
)
from prompts.reverse_translation.identity import (
    reverse_identity_prompt,
    reverse_zero_shot_prompt
)
from prompts.speaking.identity import (
    identity_prompt as speaking_identity_prompt,
    zero_shot_prompt as speaking_zero_shot_prompt
)

from prompts.translation.letter_substitutions import (
    letter_to_word_with_dot_prompt,
    dot_between_chars,  # kept as-is to match your existing import/name
    letter_to_poem_first_letter_prompt,
)
from prompts.reverse_translation.letter_substitutions import (
    reverse_letter_to_word_with_dot_prompt,
    reverse_dot_between_chars,
    reverse_letter_to_poem_first_letter_prompt,
)
from prompts.speaking.letter_substitutions import (
    letter_to_word_with_dot_prompt as speaking_letter_to_word_with_dot_prompt,
    dot_between_chars as speaking_dot_between_chars_prompt,
)

# Prompts for classic ciphers
from prompts.translation.ciphers import (
    rot13_cipher_prompt,
    base64_cipher_prompt,
    base64_2x_cipher_prompt,
    base64_3x_cipher_prompt,
    caesar_cipher_prompt,
)
from prompts.reverse_translation.ciphers import (
    reverse_rot13_cipher_prompt,
    reverse_base64_cipher_prompt,
    reverse_base64_2x_cipher_prompt,
    reverse_base64_3x_cipher_prompt,
    reverse_caesar_cipher_prompt,
)

# Prompts for compression-based transforms
from prompts.translation.compression import (
    gzip_to_bpe_encoded_prompt,
    gzip_to_base64_encoded_prompt,
)
from prompts.reverse_translation.compression import (
    reverse_gzip_to_bpe_encoded_prompt,
    reverse_gzip_to_base64_encoded_prompt,
)

# Prompts for letter permutations
from prompts.translation.letter_permutations import (
    reverse_letters_in_each_word_prompt,
    random_permute_letters_in_each_word_prompt,
    swap_even_odd_letters_in_each_word_prompt,
    reverse_fibonacci_indices_in_each_word_prompt,
)
from prompts.reverse_translation.letter_permutations import (
    reverse_reverse_letters_in_each_word_prompt,
    reverse_random_permute_letters_in_each_word_prompt,
    reverse_swap_even_odd_letters_in_each_word_prompt,
    reverse_reverse_fibonacci_indices_in_each_word_prompt,
)

# Prompts for destructive mutations
from prompts.translation.destructive_mutations import (
    replace_80pct_letters_with_star_prompt,
    first_letter_of_each_word_prompt,
    first_token_of_each_word_model_tokenizer_prompt,
)
from prompts.reverse_translation.destructive_mutations import (
    reverse_replace_80pct_letters_with_star_prompt,
    reverse_first_letter_of_each_word_prompt,
    reverse_first_token_of_each_word_model_tokenizer_prompt,
)

from prompts.speaking.letter_substitutions import (
    letter_to_word_with_dot_prompt as speaking_letter_to_word_with_dot_prompt,
)

# Translations
from prompts.speaking.translations import (
    translate_to_French as speaking_translate_to_French,
    translate_to_Chinese as speaking_translate_to_Chinese,
    translate_to_Korean as speaking_translate_to_Korean,
    translate_to_Russian as speaking_translate_to_Russian,
    translate_to_Arabic as speaking_translate_to_Arabic,
    translate_to_Adyghe as speaking_translate_to_Adyghe,
)

# Steg
from prompts.speaking.steganography import (
    speaking_math_safety_steg,
    speaking_math_sonnet_steg,
    speaking_math_news_article_steg,
    speaking_math_enterprise_java_steg,
    speaking_math_weather_report_steg,
    speaking_math_numbers_sequence_steg
)

# ---- Central registry ----

PROMPT_MAP = {
    # substitutions / identity
    "zero_shot": zero_shot_prompt,
    "reverse_zero_shot": reverse_zero_shot_prompt,
    "speaking_zero_shot": speaking_zero_shot_prompt,

    "identity": identity_prompt,
    "reverse_identity": reverse_identity_prompt,
    "speaking_identity": speaking_identity_prompt,

    "letter_to_word_with_dot": letter_to_word_with_dot_prompt,
    "reverse_letter_to_word_with_dot": reverse_letter_to_word_with_dot_prompt,
    "speaking_letter_to_word_with_dot": speaking_letter_to_word_with_dot_prompt,

    "dot_between_chars": dot_between_chars,
    "reverse_dot_between_chars": reverse_dot_between_chars,
    "speaking_dot_between_chars": speaking_dot_between_chars_prompt,

    "letter_to_poem_first_letter": letter_to_poem_first_letter_prompt,
    "reverse_letter_to_poem_first_letter": reverse_letter_to_poem_first_letter_prompt,

    # ciphers
    "rot13_cipher": rot13_cipher_prompt,
    "reverse_rot13_cipher": reverse_rot13_cipher_prompt,

    "base64_cipher": base64_cipher_prompt,
    "reverse_base64_cipher": reverse_base64_cipher_prompt,

    "base64_2x_cipher": base64_2x_cipher_prompt,
    "reverse_base64_2x_cipher": reverse_base64_2x_cipher_prompt,

    "base64_3x_cipher": base64_3x_cipher_prompt,
    "reverse_base64_3x_cipher": reverse_base64_3x_cipher_prompt,

    "caesar_cipher": caesar_cipher_prompt,
    "reverse_caesar_cipher": reverse_caesar_cipher_prompt,

    # compression
    "gzip_to_bpe_encoded": gzip_to_bpe_encoded_prompt,
    "reverse_gzip_to_bpe_encoded": reverse_gzip_to_bpe_encoded_prompt,

    "gzip_to_base64_encoded": gzip_to_base64_encoded_prompt,
    "reverse_gzip_to_base64_encoded": reverse_gzip_to_base64_encoded_prompt,

    # letter permutations
    "reverse_letters_in_each_word": reverse_letters_in_each_word_prompt,
    "reverse_reverse_letters_in_each_word": reverse_reverse_letters_in_each_word_prompt,

    "random_permute_letters_in_each_word": random_permute_letters_in_each_word_prompt,
    "reverse_random_permute_letters_in_each_word": reverse_random_permute_letters_in_each_word_prompt,

    "swap_even_odd_letters_in_each_word": swap_even_odd_letters_in_each_word_prompt,
    "reverse_swap_even_odd_letters_in_each_word": reverse_swap_even_odd_letters_in_each_word_prompt,

    "reverse_fibonacci_indices_in_each_word": reverse_fibonacci_indices_in_each_word_prompt,
    "reverse_reverse_fibonacci_indices_in_each_word": reverse_reverse_fibonacci_indices_in_each_word_prompt,
    # destructive mutations
    "replace_80pct_letters_with_star": replace_80pct_letters_with_star_prompt,
    "reverse_replace_80pct_letters_with_star": reverse_replace_80pct_letters_with_star_prompt,

    "first_letter_of_each_word": first_letter_of_each_word_prompt,
    "reverse_first_letter_of_each_word": reverse_first_letter_of_each_word_prompt,
    
    "first_token_of_each_word_model_tokenizer": first_token_of_each_word_model_tokenizer_prompt,
    "reverse_first_token_of_each_word_model_tokenizer": reverse_first_token_of_each_word_model_tokenizer_prompt,

    "speaking_French": speaking_translate_to_French,
    "speaking_Chinese": speaking_translate_to_Chinese,
    "speaking_Korean": speaking_translate_to_Korean,
    "speaking_Russian": speaking_translate_to_Russian,
    "speaking_Arabic": speaking_translate_to_Arabic,
    "speaking_Adyghe": speaking_translate_to_Adyghe,

    # steg
    "speaking_math_safety_steg": speaking_math_safety_steg,
    "speaking_math_sonnet_steg": speaking_math_sonnet_steg,
    "speaking_math_news_article_steg": speaking_math_news_article_steg,
    "speaking_math_enterprise_java_steg": speaking_math_enterprise_java_steg,
    "speaking_math_weather_report_steg": speaking_math_weather_report_steg,
    "speaking_math_numbers_sequence_steg": speaking_math_numbers_sequence_steg
}


def get_translation_prompt(encoding_scheme_name: str):
    try:
        return PROMPT_MAP[encoding_scheme_name]
    except KeyError:
        # Friendly error with suggestions
        candidates = get_close_matches(encoding_scheme_name, PROMPT_MAP.keys(), n=5, cutoff=0.6)
        suggestion = f" Did you mean: {', '.join(candidates)}?" if candidates else ""
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme_name}.{suggestion}")
