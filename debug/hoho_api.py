from IPython.display import display, Markdown
from redlines import Redlines

text = f"""
Got this for my daughter for her birthday cuz she keeps taking \
mine from my room.  Yes, adults also like pandas too.  She takes \
it everywhere with her, and it's super soft and cute.  One of the \
ears is a bit lower than the other, and I don't think that was \
designed to be asymmetrical. It's a bit small for what I paid for it \
though. 我想那一定有其他好 other options that are bigger for \
the same price.  It arrived a day earlier than expected, so I got \
to play with it myself before I gave it to my daughter.
"""

text2 = f"""
Got this for my daughter for her birthday cuz she keeps taking \
mine from my room.  Sure, adults also like these pandas too.  She takes \
it everywhere with her, and it's super soft and cute.  One of the \
ears is a bit lower than the other, and I don't think that was \
designed to be asymmetrical. It's a bit small for what I paid for it \
though. 我想那一定是 other options that are bigger for \
the same price.  It arrived a day earlier than expected, so I got \
to play with it myself before I gave it to my daughter.
"""

diff = Redlines(text, text2)
# print(Markdown(text))
# print(diff.output_markdown)
display(Markdown(diff.output_markdown))