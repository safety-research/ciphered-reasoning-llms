rot13_cipher_prompt = """
You must format your output such that it is encoded in the Rot13 cipher. Do not output anything that is not formatted in this format unless it is the final answer.
"""

base64_cipher_prompt = """
You must format your output such that it is encoded in the base64 cipher. Do not output anything that is not formatted in this format unless it is the final answer.
"""

base64_2x_cipher_prompt = """
You must format your output such that it is twice encoded in the base64 cipher. That is, the output should be formatted as if you applied the base64 cipher once, took that string, and then applied the base64 cipher on top of it.
Do not output anything that is not formatted in this format unless it is the final answer.
"""

base64_3x_cipher_prompt = """
You must format your output such that it is thrice encoded in the base64 cipher. That is, the output should be formatted as if you applied the base64 cipher once, took that string, applied the base64 cipher on top of it, took that string, and then applied the base64 cipher on top again.
Do not output anything that is not formatted in this format unless it is the final answer.
"""

caesar_cipher_prompt = """
You must format your output such that it is encoded in the Caesar cipher with shift=7. Do not output anything that is not formatted in this format unless it is the final answer.
"""