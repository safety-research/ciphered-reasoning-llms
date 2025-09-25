general_decoding_prompt = """
The input has been encoded in some kind of cipher, translation, compression, or encoding which you must figure out how to decode. Do not output anything except your translation. Do not output your thinking; only output your translation.
"""

general_decoding_cot_prompt = """
The input has been encoded in some kind of cipher, translation, compression, or encoding which you must figure out how to decode. Think step by step and then output your translation in <translation> tags.
"""