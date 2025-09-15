import codecs
import base64
import re
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from encoding_schemes.letter_permutations import is_latex


def base64_valid_percentage(text: str) -> float:
    """
    Calculates the percentage of characters in the given text that are part of valid Base64 sequences.
    
    Args:
        text (str): The input string to analyze.
    
    Returns:
        float: Percentage of characters in valid Base64 sequences.
    """
    if not text:
        return 0.0

    # Base64 pattern: groups of 4 valid chars, possibly ending with '=' or '=='
    base64_pattern = re.compile(r'(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?')

    total_length = len(text)
    valid_chars = 0

    for match in base64_pattern.finditer(text):
        candidate = match.group(0)
        if not candidate:
            continue
        try:
            # Try decoding to verify it's valid base64
            base64.b64decode(candidate, validate=True)
            valid_chars += len(candidate)
        except Exception:
            continue

    return valid_chars / total_length


_BASE64_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")



def rot13_cipher(s):
    return codecs.encode(s, "rot_13")


def inverse_rot13_cipher(s):
    return codecs.decode(s, "rot_13")


def rot13_no_math_expressions_cipher(s):
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

        l_words[i] = codecs.encode(l_words[i], "rot_13")

    return " ".join(l_words)


def base64_cipher(s):
    return base64.b64encode(s.encode("utf-8")).decode("utf-8")


def inverse_base64_cipher(s):
    """
    Best-effort Base64 decoder.
    - Discards invalid characters.
    - Auto-pads as needed.
    - Skips past malformed regions and continues decoding.
    Returns a Unicode string; undecodable bytes become '�'.
    """
    # 1) Keep only valid Base64 symbols
    filtered = "".join(c for c in s if c in _BASE64_ALPHABET)

    # Fast path: try decoding the whole thing after padding to /4
    whole = _decode_with_padding(filtered)
    if whole is not None:
        return whole.decode("utf-8", errors="replace")

    # 2) Fallback: decode in sliding 4-char windows,
    #    skipping over bad quartets so we salvage as much as possible.
    out = bytearray()
    i = 0
    n = len(filtered)
    while i < n:
        # Take up to 4 chars; pad to length 4
        chunk = filtered[i:i+4]
        if not chunk:
            break
        padded = _pad_to_quad(chunk)
        try:
            out.extend(base64.b64decode(padded, validate=False))
            i += 4  # good chunk -> advance by a full quartet
        except Exception:
            i += 1  # bad chunk -> skip one char and try again
    return out.decode("utf-8", errors="replace")


def calculate_base64_cipher_adherence(s):
    return base64_valid_percentage(s) >= 0.8


def _pad_to_quad(t: str) -> str:
    """Pad with '=' so length is a multiple of 4 (without stripping existing '=')."""
    need = (-len(t)) % 4
    return t + ("=" * need if need else "")

def _decode_with_padding(t: str):
    """Try decoding an entire cleaned string with padding; return bytes or None."""
    try:
        return base64.b64decode(_pad_to_quad(t), validate=False)
    except Exception:
        return None


def base64_2x_cipher(s):
    return base64_cipher(base64_cipher(s))


def inverse_base64_2x_cipher(s):
    return inverse_base64_cipher(inverse_base64_cipher(s))


def calculate_base64_2x_cipher_adherence(s):
    if base64_valid_percentage(s) < 0.8:
        return False

    return calculate_base64_cipher_adherence(inverse_base64_cipher(s))


def base64_3x_cipher(s):
    return base64_cipher(base64_cipher(base64_cipher(s)))


def inverse_base64_3x_cipher(s):
    return inverse_base64_cipher(inverse_base64_cipher(inverse_base64_cipher(s)))


def calculate_base64_3x_cipher_adherence(s):
    if base64_valid_percentage(s) < 0.8:
        return False

    s = inverse_base64_cipher(s)
    if base64_valid_percentage(s) < 0.8:
        return False

    return calculate_base64_cipher_adherence(inverse_base64_cipher(s))


def caesar_cipher(s):
    """
    Encode/decode a string using Caesar cipher with the given shift.
    Positive shift encodes, negative shift decodes.
    """
    shift = 7

    result = []
    for char in s:
        if "a" <= char <= "z":
            # Lowercase letters
            result.append(chr((ord(char) - ord("a") + shift) % 26 + ord("a")))
        elif "A" <= char <= "Z":
            # Uppercase letters
            result.append(chr((ord(char) - ord("A") + shift) % 26 + ord("A")))
        else:
            # Non-alphabetic characters stay the same
            result.append(char)
    return "".join(result)


def inverse_caesar_cipher(s):
    shift = -7

    result = []
    for char in s:
        if "a" <= char <= "z":
            # Lowercase letters
            result.append(chr((ord(char) - ord("a") + shift) % 26 + ord("a")))
        elif "A" <= char <= "Z":
            # Uppercase letters
            result.append(chr((ord(char) - ord("A") + shift) % 26 + ord("A")))
        else:
            # Non-alphabetic characters stay the same
            result.append(char)
    return "".join(result)