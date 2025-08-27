import re
import os
from openai import AsyncOpenAI
from asyncio import Semaphore
import asyncio


client = AsyncOpenAI(
    api_key=os.environ['ANTHROPIC_API_KEY'],
    base_url="https://api.anthropic.com/v1/",
)

rate_limit = Semaphore(30)


async def do_translation(s, language, additional_instructions):
    for i in range(200):
        try:
            async with rate_limit:
                resp = await client.chat.completions.create(
                    model="claude-sonnet-4-20250514",
                    messages=[{
                        "role": "user",
                        "content": f"""
Translate the following text to {language}. Output your translation in <translation> tags. Any content in \\boxed{{}} should be kept as is (along with the boxed LaTeX itself).
{additional_instructions}
<text>
{s}
</text>
                        """.strip()
                    }],
                    temperature=1.0,
                    max_tokens=30000
                )

                ret = resp.choices[0].message.content

                print(s)
                print(ret)

                result = re.search("<translation>(.*?)</translation>", ret, re.DOTALL)
                if not result:
                    return "askdlfjlkadsjflajdklf"
                result = result.group(1)

                return result

        except Exception as e:
            print(e)
            await asyncio.sleep(3)

    raise Exception("Reached max tries!")


async def translate_to_English(s, source_language):
    return await do_translation(s, "English", f"The text is in {source_language}. If some parts don't make sense, translate them as is. Do not try to add your own interpretation of what you think the text is trying to say.")


async def translate_to_French(s):
    return await do_translation(s, "French", "")


async def translate_to_Chinese(s):
    return await do_translation(s, "Chinese", "")


async def translate_to_Korean(s):
    return await do_translation(s, "Korean", "")


async def translate_to_Russian(s):
    return await do_translation(s, "Russian", "")


async def translate_to_Arabic(s):
    # right to left
    return await do_translation(s, "Arabic", "")


async def translate_to_Adyghe(s):
    # rare language
    return await do_translation(s, "Adyghe", "Try your best to translate any words which do not have literal translations, using loan words or your best guess. You must translate the text.")
