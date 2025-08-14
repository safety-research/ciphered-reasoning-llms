reverse_replace_80pct_letters_with_star_prompt = """
The input has been formatted such that only the first 20% of the characters in each word are preserved, and the other 80% is removed and replaced with stars of the same number of characters.
For example, the word "unexpectedly" has 12 characters, so the encoding keeps 2 characters and replaces the other 10 giving "un**********".
Do not output anything except your translation.
"""

reverse_first_letter_of_each_word_prompt = """
The input has been formatted such that only the first letter of each word is preserved, and all other characters are deleted.
For example, the word "word" has been converted to "w".
Do not output anything except your translation.
"""

reverse_first_token_of_each_word_model_tokenizer_prompt = """
The input has been formatted such that only the first token of each word (as you would see it) is preserved, and all other characters are deleted.
Do not output anything except your translation.
"""
