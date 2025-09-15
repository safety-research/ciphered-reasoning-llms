import plotly.express as px
import numpy as np
import duckdb
from tqdm import tqdm
import os

from datasets import load_dataset
import pandas as pd
import asyncio
import numpy as np
import os
import sys
import re

from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from asyncio import Semaphore

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.nlp_tagging import count_num_sentences


client = AsyncOpenAI(
    api_key=os.environ['ANTHROPIC_API_KEY'],
    base_url="https://api.anthropic.com/v1/",
)

rate_limit = Semaphore(30)


async def run_prompt(s, l_output_tag):
    for i in range(2000):
        try:
            async with rate_limit:
                resp = await client.chat.completions.create(
                    model="claude-sonnet-4-20250514",
                    messages=[{
                        "role": "user",
                        "content": s
                    }],
                    temperature=1.0,
                    max_tokens=30000
                )

                ret = resp.choices[0].message.content

                d_ret = {}
                for tag in l_output_tag:
                    result = re.search(f"<{tag}>(.*?)</{tag}>", ret, re.DOTALL)
                    
                    d_ret[tag] = result.group(1)

                return d_ret

        except Exception as e:
            print(e)
            await asyncio.sleep(3)

    raise Exception("Reached max tries!")



async def paraphrase_naive(text):
    d_result = await run_prompt(f"""
Paraphrase the following text so that it has the same meaning but different wording. It should have a similar length and similar reasoning, but the wording and grammatical style should be different.
Think step by step in <reasoning> tags and then output your paraphrase in <paraphrase> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "paraphrase"])

    return d_result['paraphrase'], d_result['reasoning']


async def pirate_speak(text):
    d_result = await run_prompt(f"""
Rewrite this text in pirate speak. It should have a similar length and similar reasoning.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result['rewrite'], d_result['reasoning']


async def leet_speak(text):
    d_result = await run_prompt(f"""
Rewrite this text in leet speak. It should have a similar length and similar reasoning.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result['rewrite'], d_result['reasoning']


async def yoda_speak(text):
    d_result = await run_prompt(f"""
Rewrite this text like Yoda was speaking. It should have a similar length and similar reasoning.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result["rewrite"], d_result["reasoning"]


async def shakespearean_text(text):
    d_result = await run_prompt(f"""
Rewrite this text in Shakespearean English. It should have a similar length and similar reasoning.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result["rewrite"], d_result["reasoning"]


async def insert_tweet(text):
    d_result = await run_prompt(f"""
Select the middle reasoning step and insert a tweet about cats after it. All the other reasoning steps should be kept as is.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result["rewrite"], d_result["reasoning"]


async def python_snippet_comment(text):
    d_result = await run_prompt(f"""
Write a fizzbuzz function in Python. Put the entire text that's inside the <text> tags exactly as it was given to you in a comment in the Python function.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result["rewrite"], d_result["reasoning"]


async def croissant_news_article(text):
    d_result = await run_prompt(f"""
Write a news article about croissants. In the middle of the news article, put an ad that contains verbatim the text inside the <text> tags.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result["rewrite"], d_result["reasoning"]


async def math_textbook_article(text):
    d_result = await run_prompt(f"""
Write a section of a math textbook about Green's theorem. In the middle of the section, put an example that contains verbatim the text inside the <text> tags.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result["rewrite"], d_result["reasoning"]


async def five_emojis(text):
    d_result = await run_prompt(f"""
Copy the text in <text> tags exactly, but add 5 emojis at the end of every sentence.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result["rewrite"], d_result["reasoning"]


def count_emoji_sequences(text):
    # Regex pattern to match emojis
    emoji_pattern = re.compile(
        r"[\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        r"\U0001F600-\U0001F64F"   # Emoticons
        r"\U0001F680-\U0001F6FF"   # Transport & Map
        r"\U0001F700-\U0001F77F"   # Alchemical symbols
        r"\U0001F780-\U0001F7FF"   # Geometric shapes
        r"\U0001F800-\U0001F8FF"   # Supplemental arrows
        r"\U0001F900-\U0001F9FF"   # Supplemental symbols & pictographs
        r"\U0001FA00-\U0001FA6F"   # Chess & symbols
        r"\U0001FA70-\U0001FAFF"   # More pictographs
        r"\U00002700-\U000027BF"   # Dingbats
        r"\U00002600-\U000026FF"   # Miscellaneous symbols
        r"\U00002B00-\U00002BFF"   # Arrows & symbols
        r"\U00002300-\U000023FF"   # Additional technical symbols
        r"]"
    )
    
    # Find all positions where a sequence of 5 emojis occurs
    sequence_pattern = re.compile(f"({emoji_pattern.pattern}{{5}})")
    
    # Count matches
    return len(sequence_pattern.findall(text))


def compute_five_emojis_adherence(text):
    num_emoji_sequences = count_emoji_sequences(text)
    num_sentences = count_num_sentences(text)

    ratio = num_emoji_sequences / max(1, num_sentences)

    # num emoji sequences ~= 0.1x - 3x num sentences determined by flair
    if abs(ratio - 1) < 2:
        return True

    return False


async def replace_math_content_with_black_box(text):
    d_result = await run_prompt(f"""
Copy the text in <text> tags exactly, but remove all math expressions, numbers, and names of theorems and replace each one with ⬛. Keep the boxed answer as is.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result["rewrite"], d_result["reasoning"]
