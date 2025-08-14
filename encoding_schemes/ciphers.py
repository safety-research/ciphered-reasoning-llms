import codecs
import base64


def rot13_cipher(s):
    return codecs.encode(s, "rot_13")


def base64_cipher(s):
    return base64.b64encode(s.encode("utf-8")).decode("utf-8")


def base64_2x_cipher(s):
    return base64_cipher(base64_cipher(s))


def base64_3x_cipher(s):
    return base64_cipher(base64_cipher(base64_cipher(s)))


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
