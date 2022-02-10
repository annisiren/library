import re

# SEARCH
text = "The person's phone number is 408-555-1234. Call soon!"

pattern = 'phone'

match = re.search(pattern,text)
match.span()
match.start()
match.end()


text = "my phone is a new phone"

match = re.search("phone",text)
match.span()

matches = re.findall("phone",text)
len(matches)

for match in re.finditer("phone",text):
    print(match.span())

match.group()

# PATTERNS
# \d	A digit	           file_\d\d	file_25
# \w	Alphanumeric	  \w-\w\w\w	    A-b_1
# \s	White space       a\sb\sc	    a b c
# \D	A non digit	      \D\D\D	    ABC
# \W	Non-alphanumeric  \W\W\W\W\W	*-+=)
# \S	Non-whitespace	  \S\S\S\S	    Yoyo

# +	    Occurs one or more times	Version \w-\w+	Version A-b1_1
# {3}	Occurs exactly 3 times	    \D{3}	        abc
# {2,4}	Occurs 2 to 4 times	        \d{2,4}	        123
# {3,}	Occurs 3 or more	        \w{3,}	        anycharacters
# \*	Occurs zero or more times	A\*B\*C*	    AAACC
# ?	    Once or none	            plurals?	    plural

text = "My telephone number is 408-555-1234"
phone = re.search(r'\d\d\d-\d\d\d-\d\d\d\d',text)
phone.group()
re.search(r'\d{3}-\d{3}-\d{4}',text)

phone_pattern = re.compile(r'(\d{3})-(\d{3})-(\d{4})')
results = re.search(phone_pattern,text)
# The entire result
results.group()
results.group(1)
results.group(2)
results.group(3)

# SYNTAX
# | operator
re.search(r"man|woman","This man was here.")
re.search(r"man|woman","This woman was here.")

# WILDCARD
re.findall(r".at","The cat in the hat sat here.")
re.findall(r".at","The bat went splat")
re.findall(r"...at","The bat went splat")

# One or more non-whitespace that ends with 'at'
re.findall(r'\S+at',"The bat went splat")

# Ends with a number
re.findall(r'\d$','This ends with a number 2')
# Starts with a number
re.findall(r'^\d','1 is the loneliest number.')

# EXCLUSION
phrase = "there are 3 numbers 34 inside 5 this sentence."

re.findall(r'[^\d]',phrase)
re.findall(r'[^\d]+',phrase)

test_phrase = 'This is a string! But it has punctuation. How can we remove it?'
re.findall('[^!.? ]+',test_phrase)

clean = ' '.join(re.findall('[^!.? ]+',test_phrase))
clean


# BRACKETS AND GROUPING
text = 'Only find the hypen-words in this sentence. But you do not know how long-ish they are'
re.findall(r'[\w]+-[\w]+',text)

# PARENTHESIS for MULTIPLE OBJECTS
# Find words that start with cat and end with one of these options: 'fish','nap', or 'claw'
text = 'Hello, would you like some catfish?'
texttwo = "Hello, would you like to take a catnap?"
textthree = "Hello, have you seen this caterpillar?"

re.search(r'cat(fish|nap|claw)',text)
re.search(r'cat(fish|nap|claw)',texttwo)
# None returned
re.search(r'cat(fish|nap|claw)',textthree)


# https://docs.python.org/3/howto/regex.html
