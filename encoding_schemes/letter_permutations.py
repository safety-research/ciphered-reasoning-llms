import numpy as np
import json
import string
import os


def reverse_letters_in_each_word(s):
    s = s.split(" ")
    for i in range(len(s)):
        s[i] = s[i][::-1]

    return " ".join(s)


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

    words = s.split(" ")

    n_valid_words = 0
    for word in words:
        word = normalize_word(word)
        if word in english_dictionary or len(word) == 0:
            n_valid_words += 1

    return n_valid_words / len(words) >= 0.7
        

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
