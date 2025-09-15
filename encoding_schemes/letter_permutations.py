import numpy as np
import json
import string
import os
import re

def is_latex(text: str) -> bool:
    """
    Detects whether the given string contains LaTeX content.

    Args:
        text (str): The input string.

    Returns:
        bool: True if the string likely contains LaTeX, False otherwise.
    """
    # Common LaTeX patterns
    latex_patterns = [
        r"\$.*?\$",            # Inline math: $...$
        r"\$\$.*?\$\$",        # Display math: $$...$$
        r"\\\[.*?\\\]",        # Display math: \[...\]
        r"\\\(.*?\\\)",        # Inline math: \(...\)
        r"\\begin\{.*?\}",     # LaTeX environments: \begin{...}
        r"\\end\{.*?\}",       # LaTeX environments: \end{...}
        r"\\[a-zA-Z]+",        # LaTeX commands: \frac, \alpha, \sum, etc.
    ]

    combined_pattern = re.compile("|".join(latex_patterns), re.DOTALL)
    return bool(combined_pattern.search(text))


def reverse_letters_in_each_word(s):
    s = s.split(" ")
    for i in range(len(s)):
        s[i] = s[i][::-1]

    return " ".join(s)


def reverse_letters_in_each_word_no_math_expressions_cipher(s):
    l_words = s.split(" ")
    for i, word in enumerate(l_words):
        if is_latex(word):
            continue
        
        has_math_expr = False
        for math_expr_char in ["+", "-", "/", "*", "<", ">", "=", "^", "f(x)", "\\sum"]:
            if math_expr_char in word:
                has_math_expr = True
                break
        if has_math_expr:
            continue

        if word.isdigit():
            continue

        l_words[i] = l_words[i][::-1]

    return " ".join(l_words)


def get_English_dictionary():
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "english_dictionary", "words_dictionary.json"), "r") as fp:
        return set(json.load(fp).keys())


def normalize_word(word):
    word = word.lower()
    word = "".join([c for c in word if c in string.ascii_lowercase])

    return word


def calculate_letter_permutation_adherence(s, inverse_fn):
    english_dictionary = get_English_dictionary()

    decoded = inverse_fn(s)

    words = decoded.split(" ")

    n_valid_words = 0
    n_non_latex_words = 0
    for word in words:
        if is_latex(word):
            continue

        n_non_latex_words += 1

        word = normalize_word(word)
        if word in english_dictionary or len(word) == 0:
            n_valid_words += 1

    return n_valid_words / max(n_non_latex_words, 1) >= 0.7
        

def random_permute_letters_in_each_word(s):
    s = s.split(" ")
    for i in range(len(s)):
        chars = list(s[i])

        np.random.seed(42)
        np.random.shuffle(chars)

        s[i] = "".join(chars)

    return " ".join(s)


def swap_even_odd_letters_in_each_word(s):
    s = s.split(" ")
    for i in range(len(s)):
        chars = list(s[i])

        for j in range(1, len(chars), 2):
            tmp = chars[j]
            chars[j] = chars[j - 1]
            chars[j - 1] = tmp

        s[i] = "".join(chars)

    return " ".join(s)


def fib_upto(n):
    """Yield Fibonacci numbers < n."""
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


def reverse_fibonacci_indices_in_each_word(s):
    words = s.split(" ")
    for i, w in enumerate(words):
        chars = list(w)

        # Get Fibonacci indices < len(chars)
        fib_idxs = sorted(set(j for j in fib_upto(len(chars)) if 0 <= j < len(chars)))
        if not fib_idxs:
            continue

        rev = fib_idxs[::-1]
        half = len(fib_idxs) // 2

        # swap corresponding indices
        for k in range(half):
            a = fib_idxs[k]
            b = rev[k]
            chars[a], chars[b] = chars[b], chars[a]

        words[i] = "".join(chars)

    return " ".join(words)
