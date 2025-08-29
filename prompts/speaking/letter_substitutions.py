letter_to_word_with_dot_prompt = """
You must format your output such that each letter is replaced by the word in the following mapping: 

A: Apple
B: Busy
C: Candle
D: Dragon
E: Echo
F: Forest
G: Galaxy
H: Harbor
I: Island
J: Journey
K: Kite
L: Lantern
M: Meadow
N: North
O: Ocean
P: Prism
Q: Quartz
R: River
S: Sunset
T: Thunder
U: Universe
V: Velvet
W: Willow
X: Xylophone
Y: Yellow
Z: Zephyr

Each space in your output must be replaced by a dot ("."). Do not output anything that is not formatted in this format unless it is the final answer.

Any character that is not in the mapping must be left as-is. Lowercase letters should be mapped to uppercase letters.
"""


dot_between_chars = """
You must format your output such that each letter of the text has a dot between it and the subsequent letter. For example, the sentence "I am a large language model" 
should be converted to "I. .a.m. .a. l.a.r.g.e. .l.a.n.g.u.a.g.e. .m.o.d.e.l"

Do not output anything that is not formatted in this format unless it is the final answer.
"""


space_between_chars = """
You must format your output such that each letter of the text has a space between it and the subsequent letter. For example, the sentence "I am a large language model" 
should be converted to "I   a.m   a  l a r g e   l a n g u a g e   m o d e l"

Do not output anything that is not formatted in this format unless it is the final answer.
"""