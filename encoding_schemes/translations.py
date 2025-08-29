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


briefhand_instructions = """
<info>
Briefhand
Introduction
Briefhand, also known as Carter Briefhand, is a shorthand (abbreviated writing) system that uses the standard A-to-Z alphabet and normal punctuation marks. Briefhand does not use any unique symbols. It was taught in a few American schools in the 1957-1980 period and was proven to be relatively easy to learn.

Copyright Note: In a 1931 case (Brief English Systems v. Owen, 48 F.2d 555) a United States appeals court ruled that a particular textbook describing a shorthand system can be copyrighted but the system itself is an invention or a process rather than a literary work and cannot be copyrighted. Other authors would be free to write a new textbook of the system or to create a very similar system. This description of Briefhand is presented in light of that precedent.

History
Theodore Hampton Carter (listed as principle author of the Briefhand book) was head of the Business Administration Department at Central High School in Providence, Rhode Island when the first Briefhand textbooks were originally published. I have not been able to find any other biographical information about him.

Briefhand appeared in the late 1950s, a time when school systems were gradually losing interest in symbol-based shorthand and were experimenting with many alphabet-based systems. Although some shorthand instructors enthusiastically promoted Briefhand, it faced competition from the publishers and promoters of Forkner, Abbreviatrix, Zinman’s Rapid Writing, assorted variants of Speedwriting and a dozen other systems.

In the early 1970s Carl Salser and Theo Yerian began publishing textbooks of “Personal Shorthand” which is nearly identical to Briefhand. (Yerian was listed as a co-author of the original Briefhand textbook).

Proof of Usefulness
In The Journal of Business Education (Volume 35, 1960 - Issue 7) an article entitled Rx Briefhand? by Pauline J. Oliver describes the results of Briefhand classes at Portland State College. It states, “In one term, students can triple their longhand writing speed.” Nearly all students achieved a speed of 60 wpm and about half of all students reached 80 wpm. The dropout rate of Briefhand classes was very low compared to classes in symbol-based shorthand systems.

Find Available Textbooks
There are two versions of the Briefhand textbook: the Complete Edition with 295 pages and a smaller Basic Edition. (There was also a dictionary, a workbook and a teacher’s guide but those are rare and you probably won’t to be able to obtain them.) Here are ready-to-use searches for the textbook.

ABEbooks.com search
Amazon search number 1
Amazon search number 2
eBay search

You can find Personal Shorthand textbooks by doing an advanced search for title=shorthand and author=salser at sites like ABEBooks and Amazon.

Five Rules of Briefhand Theory
These five rules are the core of Briefhand.

Never write a silent vowel. In Briefhand, like is written as lik and gate is written as gat

At the beginning of a word, non-silent vowels must be written. acre is written as acr and alike becomes alik (Although it is not mentioned in the Briefhand textbook’s theory rules, a non-silent vowel at the end of a word must also be written, with some exceptions.)

Non-silent vowels within a word are omitted if they are “short” vowels but “long” vowels are written:
bit : bt
bite : bit
mat : mt
mate : mat

If a word contains a doubled vowel, only one of the letters is written:
spoon : spon
seed : sed

Silent consonants in a word are omitted and doubled consonants are reduced to one.
night : nit
pillow : plo

It seems to me that students should spend some time writing a variety of words using only the five principles above before they move on to the next phases of study. Give the brain a few days to get truly comfortable with the Five Pillars of Briefhand.

Punctuation
Punctuation marks such as periods, quotes and possessive apostrophes are used in Briefhand exactly the same way they are used in regular handwriting.

In the Briefhand textbook, the first letter of each sentence and the first letter of the names of people and places is not capitalized in the usual way. Instead, each letter that we think of as capitalized is written in lower-case form and then underlined. This strikes me as remarkably inefficient. When I dabble in Briefhand I just type everything in lower-case letters. The authors of the Personal Shorthand books suggested the following:

capital letters… may be written in the normal manner; or they may be ignored entirely… Writers have no trouble remembering (when transcribing) that the initial letter in each sentence should be capitalized…

The symbol // (a pair of diagonal slashes) may be used to indicate the start of a new paragraph. On an old-fashioned manual typewriter that would have been faster than lifting your left hand off the keyboard to slap the carriage return lever.

You can save a little time by omitting almost all commas from your shorthand. This practice is common in almost all shorthand systems but the founders of Briefhand did not adopt this valuable practice; in fact the shorthand specimens in the textbook use more commas than seem normal by 21st century standards.

Phonetic Abbreviations
In Briefhand each letter of the alphabet can be used as an abbreviation for a few common syllables or prefixes/suffixes. Here are a few examples of these phonetic abbreviations. This is not a complete list.

c represents ch, -tch, com, con
rich : rc
latch : lc
compete : cpt
control : ctrol

f stands for ph, fē, for, fore, -ful, -ify
phone : fon
fever : fvr
foretell : ftl
before : bf
prideful : pridf
signify : sgnf

g represents -ing and -dge among other things
selling : slg
ledge : lg

l stands for lē, -ly, -lty, -lity
legal : lgl
slowly : slol
realty : rl
plurality : plrl

n represents nē, -nce, -ness among other things
neglect : nglk
advance : advn
sadness : sdn

r stands for rē, -rity, -ure
recurring : rcrg
rarity : rr
figure : fgr

s represents sē, sh, -sion, -tion among other things
sequence : sqn
shape : sap
occasion : ocas
ration : rs

t represents the sounds of th among other things
throne : tron
bathing : batg

w represents the sounds wē, ow, ou as in proud and aw as in saw
weasel : wsl
plow : plw
found : fwn
lawful : lwf

Notice the pattern in legal, fever, compete. The “long e” sound in some words is omitted (rather than written with an e) because there may be an implied “long e” sound after most consonants if they are not being used for any other abbreviation. Thus genie is written as gn. However, when the “long e” sound is written with ee in longhand, it is written with e in Briefhand: weed becomes wed

In contrast to Briefhand, Personal Shorthand specifies a 6th basic theory rule: a final non-silent vowel must be written unless it’s the “ee” sound preceded by a consonant. When a word ends with a consonant followed by the “ee” sound, only the consonant is written. Thus salary is written as slr and so forth. In Briefhand the theoretical explanation for this is found in the rules of phonetic abbreviations for each consonant rather than in a basic theory rule.

Brief Forms
Every practical shorthand system provides short abbreviations for the most common words. The Briefhand textbook calls these “high frequency abbreviations” is some places and “brief forms” in other spots. Here are some examples. This is not a complete list.

a stands for an, and, at, about
c represents can, come, copy, credit
e stands for the, he
g represents go, good, get, glad, give
i stands for I, it, is, time
l represents will, well, all, also, letter
m stands for am, me, my, man, men, made
n represents in, not, no, know, when, information
o stands for of, on, out, what
r represents are, our, or, order, return
s is the abbreviation for she, so, sincerely, wish, see
t stands for to, too, this, that, there, thank
u stands for you, up, under, us
v represents have, ever, every, very, receive
w stands for with, we, were, which, how, now
z represents as, has, was, his

25 of the 26 letters have multiple words assigned to each brief form. The letter x only stands for one word: “check.” Personally I use x as a brief form for “not” and thus reduce the ambiguity of the brief form n

Some students may find it helpful to compose an absurd sentence that uses all the possible meanings of each brief form. For example: “We know no information about when he was not in the hotel.” Imagine what you might say about you up under us. A little bit of silliness may firm up the memory when tedious drilling won’t work.

It is interesting to note that some brief form words are represented by their last letters:

would becomes d
if becomes f
they and why become y

A few brief forms are based on the final sound rather than the last letter:

as, has, was, his become z
are becomes r
take, make become k

Phrases
In most shorthand systems brief forms are written together without any spaces to write common phrases in a rapid and compact manner. For example, in Dearborn’s Speedwriting “will you please” is written as lupl and in Gregg shorthand the symbols avnba are connected into one outline that means “I have not been able.”

Briefhand distinctively does not connect the brief forms that are used in phrases. They are written with spaces between them. This prevents the potential ambiguity that would result from gluing brief forms together: if Briefhand didn’t preserve the spaces between brief forms, lv could mean either will have or love. Every group of letters would have to be parsed more laboriously to determine if it was a cluster of brief forms or an individual word.

Below are examples of some frequent phrases.

Briefhand	longhand	~	Briefhand	longhand
o e	of the		t e	to the
a e	and the		w e	with the
d b	would be		d n	did not
e i	he is		e z	he was
f y	for your		f u	for you
f u r	if you are		f t i	if there is
g nf	good enough		l b g	will be glad
i m	I am		i v	I have
i z b	it has been		s z b	she has been
an i	any time		a e i	at the time
n e	in the		n e	when the
nt e	into the		n dwt	no doubt
o is	of its		o i	what is
t i	there is		t r	there are
u v	you have		u n	you know
v mc	very much		w mc	how much
y r	they are		y v	they have
z n	was not		z n b	has not been
1 o e	one of the		1 o tes	one of these
 

Word Beginnings & Endings
Briefhand has shortcuts for some of the prefixes, suffixes and syllables that often occur at the beginnings and endings of words. Here are examples of some that may need a little study and practice:

Beginnings:

cau is reduced to c
ex is reduced to x
per, por-, pr, pro-, pur are reduced to p
re is reduced to r
super is reduced to spr

en is reduced to n
in is reduced to n except before another n
un is reduced to n except before another n

em is reduced to m except before vowels
im is reduced to m except before another m

for becomes f
under becomes u
with becomes w

Endings:

-ciency is written as sc
-cient becomes st
-ification is written as fcas
-less becomes ls
-ment is written as m
-ology becomes log
-ort is reduced to rt
-ship is written as sp
-tain becomes tan
-ual is reduced to ul

The past tense -ed is always written as d even though it is sometimes pronounced like /t/ : hissed > hsd

The plural -s is always written as s even though it is often pronounced like /z/ : degrees > dgres

Other endings such as -ful and -ing are listed in the Phonetic Abbreviations section.

The Briefhand and Personal Shorthand books take different approaches on the topic of word beginnings and endings. Fewer items are listed in the PS books and they emphasize that the user can write words phonetically according to the most basic rules of the system rather than using the shortcuts if that is preferred.

Sample Sentences
from Carter Briefhand (Complete Edition) but with normalized capitalization:

Japan i n e tmprt zon a is climt i v mc lik t o ars n e US lig n cprb lttuds. n gnrl, e smrs r ht a humd a e wntrs mild. e climt arwn Tokyo i ofn cprd t t o Wash. DC, to n e mtns nrtrn scss o Honshu a o Hokkaido e wntrs r svr. o e wstrn sid o tes mtns t i ofn nf sno f lg ssns o skg.

/the end/
</info>
"""


async def translate_to_Briefhand(s):
    return await do_translation(s, "Briefhand", briefhand_instructions)


async def translate_from_Briefhand(s):
    return await do_translation(s, "English", briefhand_instructions + "\nThe text is in Briefhand. If some parts don't make sense, translate them as is. Do not try to add your own interpretation of what you think the text is trying to say.")


async def translate_to_morse_code(s):
    return await do_translation(s, "Morse code", "Any characters which do not have Morse code mappings should be kept as is.")


async def translate_to_Python(s):
    return await do_translation(s, "Python without comments", "")


async def translate_to_enterprise_Java(s):
    return await do_translation(s, "enterprise Java without comments", "")