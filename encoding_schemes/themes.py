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


async def replace_math_content_with_black_box(text):
    d_result = await run_prompt(f"""
Copy the text in <text> tags exactly, but remove all math expressions, numbers, and names of theorems and replace each one with ⬛. Keep the boxed answer as is.
Think step by step in <reasoning> tags and then output your rewrite in <rewrite> tags.

<text>{text}</text>
    """.strip(), ["reasoning", "rewrite"])

    return d_result["rewrite"], d_result["reasoning"]
