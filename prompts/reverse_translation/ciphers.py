reverse_rot13_cipher_prompt = """
The input has been encoded in the Rot13 cipher. Do not output anything except your translation.
"""

reverse_base64_cipher_prompt = """
The input has been encoded in the base64 cipher. Do not output anything except your translation.
"""

reverse_base64_2x_cipher_prompt = """
The input has been twice encoded in the base64 cipher.
That is, the input is formatted as if you applied the base64 cipher once, took that string, and then applied the base64 cipher on top of it.
Do not output anything except your translation.
"""

reverse_base64_3x_cipher_prompt = """
The input has been thrice encoded in the base64 cipher.
That is, the input is formatted as if you applied the base64 cipher once, took that string, and applied the base64 cipher, then took that string and applied the base64 cipher on top again.
Do not output anything except your translation.
"""

reverse_caesar_cipher_prompt = """
The input has been encoded in the Caesar cipher with shift=7.
Do not output anything except your translation.
"""
