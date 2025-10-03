replace_80pct_letters_with_star_prompt = """
You must format your output such that only the first 20% of the characters in each word are preserved, and the other 80% is removed and replaced with stars of the same number of characters.
For example, the word "unexpectedly" has 12 characters, so you should keep 2 characters and replace the other 10 and output "un**********".
Do not output anything that is not formatted in this format unless it is the final answer.
"""

first_letter_of_each_word_prompt = """
You must format your output such that only the first letter of each word is preserved, and all other characters are deleted.
For example, the word "word" should be converted to "w".
Do not output anything that is not formatted in this format unless it is the final answer.
"""

first_token_of_each_word_model_tokenizer_prompt = """
You must format your output such that only the first token of each word (as you see it) is preserved, and all other characters are deleted.
Do not output anything that is not formatted in this format unless it is the final answer.
"""


remove_all_nouns_prompt = """
Remove all nouns (including proper nouns) from your output, except the final answer. Do not output anything that is not formatted in this format unless it is the final answer.
"""

remove_all_verbs_prompt = """
Remove all verbs from your output, except the final answer. Do not output anything that is not formatted in this format unless it is the final answer.
"""
