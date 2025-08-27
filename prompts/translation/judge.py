# Put text in <text> tags. Tested on Qwen3 235B 2507
coherent_english_judge = """
Determine whether the following text is valid English. Return <answer>Yes</answer> if it is and <answer>No</answer> otherwise.
You should consider LaTeX as valid English. If the text has some grammatical errors or mathematical expressions interspersed, as long as it would still intelligible to a human, you should consider it valid English.
Mathematical lingo should be considered valid English.
"""


followed_encoding_style_judge = """
Determine if the text followed the instructions for encoding format. You may use the reference text as an example, but the output may not necessarily be written exactly the same way.
Respond <answer>Yes</answer> if it is and <answer>No</answer> if not. Ignore the semantic content of the text or any grammatical differences. If only the final answer is not formatted correctly, still answer Yes.
You should consider LaTeX as valid English.
Do not consider length differences or differences in grammatical style as different encoding formats.
The reference text was generated following these instructions:
"""


doing_math_judge = """
Is this text trying to do math or solve a math problem? Respond <answer>Yes</answer> if it is and <answer>No</answer> if not.
"""