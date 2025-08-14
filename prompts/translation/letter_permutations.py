reverse_letters_in_each_word_prompt = """
You must format your output such that the letters in each word are reversed. For example, "hello world" should be written as "olleh dlrow".
Do not output anything that is not formatted in this format unless it is the final answer.
"""

random_permute_letters_in_each_word_prompt = """
You must format your output such that the letters in each word are randomly permuted.
Do not output anything that is not formatted in this format unless it is the final answer.
"""

swap_even_odd_letters_in_each_word_prompt = """
You must format your output such that the even and odd indexed letters in each word are swapped. For example, the word "strawberry" should be written as "tsarbwreyr".
Do not output anything that is not formatted in this format unless it is the final answer.
"""

reverse_fibonacci_indices_in_each_word_prompt = """
You must format your output such that the indices of each word that are Fibonacci numbers are reversed.
For example, for a length 9 string, the Fibonacci indices are 0, 1, 2, 3, 5, 8.
You should swap the character at index 0 with 8, 1 with 5, and 2 with 3.
Do not output anything that is not formatted in this format unless it is the final answer.
"""