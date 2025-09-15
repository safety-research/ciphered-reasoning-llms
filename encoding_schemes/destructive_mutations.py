import tiktoken
from transformers import AutoTokenizer
import re
import os
import sys
from openai import AsyncOpenAI
from asyncio import Semaphore
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.nlp_tagging import count_num_nouns, count_num_verbs, count_num_sentences


def replace_80pct_letters_with_star(s):
    s = s.split(" ")

    for i in range(len(s)):
        start_idx = int(len(s) * 0.2)
        star_len = len(s[i]) - start_idx

        s[i] = s[i][:start_idx] + ("*" * star_len)

    return " ".join(s)


def first_letter_of_each_word(s):
    s = s.split(" ")

    for i in range(len(s)):
        s[i] = s[i][:1]

    return " ".join(s)


cl100k_encoding = tiktoken.get_encoding("cl100k_base")


def first_token_of_each_word_cl100k_base(s):
    ids = enc.encode_ordinary(text)

    if len(ids) == 0:
        return ""

    b = enc.decode_single_token_bytes(ids[0])
    return b.decode("utf-8")


d_tokenizer_cache = {}


def first_token_string(text: str, tokenizer: AutoTokenizer) -> str | None:
    # Encode with specials, then skip them explicitly (covers BOS/EOS etc.)
    enc = tokenizer(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    if len(tokens) == 0:
        return ""

    return tokens[0]


def first_token_of_each_word_model_tokenizer(s, model):
    if model not in d_tokenizer_cache:
        d_tokenizer_cache[model] = AutoTokenizer.from_pretrained(model)

    tokenizer = d_tokenizer_cache[model]

    s = s.split(" ")

    for i in range(len(s)):
        s[i] = first_token_string(s[i], tokenizer)

    return " ".join(s)


client = AsyncOpenAI(
    api_key=os.environ['ANTHROPIC_API_KEY'],
    base_url="https://api.anthropic.com/v1/",
)

rate_limit = Semaphore(30)


async def run_prompt(s, search_tag):
    for i in range(200):
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

                print(s)
                print(ret)

                result = re.search(f"<{search_tag}>(.*?)</{search_tag}>", ret, re.DOTALL)
                if not result:
                    return "askdlfjlkadsjflajdklf", ret
                result = result.group(1)

                return result, ret

        except Exception as e:
            print(e)
            await asyncio.sleep(3)

    raise Exception("Reached max tries!")


async def remove_all_nouns(s):
    return await run_prompt("""
    Remove all nouns (including proper nouns) from the following text and output your translation in <translation> tags. Think step by step in <reasoning> tags before outputting your translation. Keep anything \\boxed{} as is.
    """ + "\n" + f"<text>{s}</text>", "translation")


async def remove_all_verbs(s):
    return await run_prompt("""
    Remove all verbs from the following text and output your translation in <translation> tags. Think step by step in <reasoning> tags before outputting your translation. Keep anything \boxed{} as is.
    """ + "\n" + f"<text>{s}</text>", "translation")


def calculate_remove_all_verbs_adherence(s):
    # max 20% of sentences have a verb to account for false positives
    return count_num_verbs(s, ignore_sentence_if_found=["boxed", "draw"]) <= (count_num_sentences(s) / 5)


def calculate_remove_all_nouns_adherence(s):
    # max 20% of sentences have a verb to account for false positives
    return count_num_nouns(s, ignore_sentence_if_found=["boxed", "draw"]) <= (count_num_sentences(s) / 5)


def calculate_zero_shot_adherence(s):
    search_result = re.search(r"(.*?)(\\boxed{.*?})(.*?)", s, re.DOTALL)
    if not search_result:
        return False

    remaining_len = len(search_result.group(1)) + len(search_result.group(3))

    return (remaining_len / len(s)) < 0.5
